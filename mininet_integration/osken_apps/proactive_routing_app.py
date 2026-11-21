"""os-ken (maintained Ryu fork) app: proactive shortest-path routing + event log.

Run by ``controller_manager.py`` as::

    osken-manager --ofp-tcp-listen-port 6653 proactive_routing_app.py

Environment variables:
    MININET_SDN_SPEC          path to the topology spec JSON
    MININET_SDN_CONTROLLER_ID simulator controller id (1..5)
    MININET_SDN_EVENT_LOG     JSONL file receiving timing events

Every controller installs the *same* destination-IP routes, so moving a switch
between controllers re-installs identical flows (OFPFC_ADD overwrites) without
interrupting the data plane. The table-miss entry drops packets, so nothing is
ever flooded on the cyclic OS3E graph and no packet-in storm can occur.

Logged events (wall-clock ``t`` from time.time()):
    controller_started, switch_connected, flows_ready (after BarrierReply),
    switch_disconnected
"""

import json
import os
import time

from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import (
    CONFIG_DISPATCHER,
    DEAD_DISPATCHER,
    MAIN_DISPATCHER,
    set_ev_cls,
)
from os_ken.ofproto import ofproto_v1_3

ROUTE_PRIORITY = 100
TABLE_MISS_PRIORITY = 0
ETH_TYPE_IPV4 = 0x0800


class ProactiveRoutingApp(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller_id = int(os.environ.get("MININET_SDN_CONTROLLER_ID", "0"))
        self.event_log_path = os.environ.get("MININET_SDN_EVENT_LOG")
        spec_path = os.environ["MININET_SDN_SPEC"]
        with open(spec_path, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
        self.dpid_to_index = {int(sw["dpid"]): int(sw["index"]) for sw in spec["switches"]}
        self.routes = {int(k): v for k, v in spec["routes"].items()}
        self.pending = {}  # xid -> (dpid, t_connected, flow_count)
        self._log("controller_started", pid=os.getpid())

    def _log(self, event, **fields):
        record = {"event": event, "t": time.time(), "controller_id": self.controller_id}
        record.update(fields)
        if self.event_log_path:
            with open(self.event_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
        self.logger.info("%s", record)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        dpid = int(datapath.id)
        t_connected = time.time()
        self._log("switch_connected", dpid=dpid)

        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        index = self.dpid_to_index.get(dpid)
        flows = 0

        # Table-miss: drop (empty instruction list).
        datapath.send_msg(
            parser.OFPFlowMod(
                datapath=datapath,
                priority=TABLE_MISS_PRIORITY,
                match=parser.OFPMatch(),
                instructions=[],
                command=ofproto.OFPFC_ADD,
            )
        )
        flows += 1

        if index is not None:
            for dst_ip, out_port in sorted(self.routes.get(index, {}).items()):
                match = parser.OFPMatch(eth_type=ETH_TYPE_IPV4, ipv4_dst=dst_ip)
                actions = [parser.OFPActionOutput(int(out_port))]
                inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
                datapath.send_msg(
                    parser.OFPFlowMod(
                        datapath=datapath,
                        priority=ROUTE_PRIORITY,
                        match=match,
                        instructions=inst,
                        command=ofproto.OFPFC_ADD,
                    )
                )
                flows += 1
        else:
            self._log("unknown_dpid", dpid=dpid)

        barrier = parser.OFPBarrierRequest(datapath)
        datapath.set_xid(barrier)
        self.pending[barrier.xid] = (dpid, t_connected, flows)
        datapath.send_msg(barrier)

    @set_ev_cls(ofp_event.EventOFPBarrierReply, [CONFIG_DISPATCHER, MAIN_DISPATCHER])
    def barrier_reply_handler(self, ev):
        entry = self.pending.pop(ev.msg.xid, None)
        if entry is None:
            return
        dpid, t_connected, flows = entry
        now = time.time()
        self._log(
            "flows_ready",
            dpid=dpid,
            flows=flows,
            flow_install_ms=(now - t_connected) * 1000.0,
        )

    @set_ev_cls(ofp_event.EventOFPStateChange, [DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        datapath = ev.datapath
        if datapath is not None and datapath.id is not None:
            self._log("switch_disconnected", dpid=int(datapath.id))
