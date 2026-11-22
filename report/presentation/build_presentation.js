// Final presentation deck (CSC406S6 / CSC406M6, viva 1 December 2026).
// Build:  cd report/presentation && node build_presentation.js
// QA view of the reserved webcam corner:  SHOW_CAM=1 node build_presentation.js
//
// The top-right corner (CAM) is left empty on every slide because the presenter's
// face must stay visible there throughout the recorded video.

const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const SKILL = "/home/dharshan/.claude/skills/synced/317cfba6-3be6-4c90-a5b5-bfaaba326cf1_432ac36a-50d0-4b39-9dca-660ed49f5ef5/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const FIG = path.resolve(__dirname, "../latex/figures");
const OUT = path.resolve(__dirname, "FINAL_PRESENTATION.pptx");

const THEME = {
  name: "SDN Controller Management",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "15233B", lt1: "FFFFFF", dk2: "3D4B5F", lt2: "EEF3F8",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "5B6B7F", accent6: "0E5A8A", hlink: "2A78D6", folHlink: "0E5A8A",
  },
};
const HEX = THEME.colors;
const CAM = { x: 10.35, y: 0, w: 2.98, h: 2.3 }; // reserved webcam corner (inches)

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "M. Logadharsan";
pres.title = "DRL for Proactive Load-Balancing Controller Management in SDN";
pres.subject = "Final-year research project presentation";
const C = pres.SchemeColor;

// ------------------------------------------------------------------ layouts
const footerText = { text: "M. Logadharsan  ·  2021/SP/098  ·  DRL controller management in SDN",
  options: { x: 0.6, y: 7.05, w: 7, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } };

pres.defineSlideMaster({
  title: "DARK",
  background: { color: HEX.dk1 },
  objects: [],
  placeholders: undefined,
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [{ text: footerText }],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.35, fontSize: 10, color: C.text2, align: "right" },
});

// ------------------------------------------------------------------ helpers
async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size }));
  const png = await sharp(Buffer.from(svg)).resize(size, size).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

function title(slide, text) {
  slide.addText(text, {
    x: 0.6, y: 0.4, w: 9.5, h: 0.95, fontFace: THEME.headFontFace, fontSize: 32, bold: true,
    color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "Title",
  });
}

function camGuide(slide) {
  if (!process.env.SHOW_CAM) return;
  slide.addShape(pres.shapes.RECTANGLE, { ...CAM, fill: { color: "FF0000", transparency: 70 },
    line: { color: "FF0000" }, objectName: "CAM-QA" });
}

async function iconCircle(slide, Comp, x, y, d, bg, fg, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: bg }, line: { color: bg }, objectName: name + " circle" });
  const pad = d * 0.24;
  slide.addImage({ data: await icon(Comp, fg), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad, objectName: name + " icon", altText: name });
}

function card(slide, x, y, w, h, fill, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill },
    line: { color: fill }, objectName: name });
}

function stat(slide, x, y, w, big, small, color, name) {
  slide.addText(big, { x, y, w, h: 0.75, fontFace: THEME.headFontFace, fontSize: 36, bold: true, color,
    margin: 0, isTextBox: true, objectName: name + " value" });
  slide.addText(small, { x, y: y + 0.75, w, h: 0.62, fontSize: 14, color: C.text2, margin: 0,
    valign: "top", isTextBox: true, objectName: name + " label" });
}

const chartBase = () => ({
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  titleFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  valGridLine: { color: "E4E3DF", size: 0.75 }, catGridLine: { style: "none" },
  showValue: true, dataLabelFontSize: 11, dataLabelColor: HEX.dk1,
  showTitle: true, titleFontSize: 14, titleColor: HEX.dk1,
});

// ------------------------------------------------------------------ slides
(async () => {
  // ============ 1. Title ============
  pres.addSection({ title: "Title" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Title" });
  s.addText("Final-Year Research Project  ·  CSC406S6 / CSC406M6", { x: 0.6, y: 0.7, w: 9.2, h: 0.4,
    fontSize: 16, color: HEX.accent4, margin: 0, isTextBox: true, objectName: "Course" });
  s.addText("A Deep Reinforcement Learning Approach for Proactive Load-Balancing Controller Management in Software-Defined Networks",
    { x: 0.6, y: 1.3, w: 9.3, h: 2.3, fontFace: THEME.headFontFace, fontSize: 34, bold: true, color: HEX.lt1,
      valign: "top", margin: 0, isTextBox: true, objectName: "Title" });
  s.addText([
    { text: "M. Logadharsan", options: { bold: true, fontSize: 22, breakLine: true } },
    { text: "Registration number: 2021/SP/098", options: { fontSize: 16, breakLine: true } },
    { text: "Supervisor: Dr. K. Thabotharan", options: { fontSize: 16, breakLine: true } },
    { text: "Department of Computer Science, University of Jaffna  ·  1 December 2026", options: { fontSize: 14, color: "C9D6E8" } },
  ], { x: 0.6, y: 4.35, w: 8.0, h: 2.0, color: HEX.lt1, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true, objectName: "Presenter" });
  // network-topology motif (bottom right, below the webcam corner)
  const nodes = [[10.0, 4.0], [11.6, 3.4], [12.6, 4.6], [11.1, 5.2], [9.6, 5.9], [12.2, 6.3], [10.7, 6.6]];
  const links = [[0, 1], [1, 2], [1, 3], [0, 3], [3, 2], [3, 4], [2, 5], [3, 6], [4, 6], [6, 5]];
  links.forEach(([a, b], i) => {
    const [x1, y1] = nodes[a], [x2, y2] = nodes[b];
    s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
      flipH: (x2 - x1) * (y2 - y1) < 0, line: { color: "5B7BA6", width: 1.5 }, objectName: `Link ${i + 1}` });
  });
  nodes.forEach(([x, y], i) => {
    const ctrl = i === 1 || i === 4 || i === 5;
    const d = ctrl ? 0.42 : 0.28;
    s.addShape(pres.shapes.OVAL, { x: x - d / 2, y: y - d / 2, w: d, h: d, fill: { color: ctrl ? HEX.accent2 : HEX.accent1 },
      line: { color: HEX.lt1, width: 1.5 }, objectName: `Node ${i + 1}` });
  });
  camGuide(s);
  s.addNotes(
`(About 25 seconds.) Good morning. I am M. Logadharsan, registration number 2021/SP/098, and my supervisor is Dr. K. Thabotharan. My project is "A Deep Reinforcement Learning Approach for Proactive Load-Balancing Controller Management in Software-Defined Networks". In short: I trained an agent that decides, while the network is running, when to move switches between controllers and when to switch controllers on or off, so that latency, energy and load stay balanced.`);

  // ============ 2. Background & motivation ============
  pres.addSection({ title: "Introduction" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  title(s, "Why controller management must adapt at runtime");
  const rows = [
    [fa.FaServer, "Distributed SDN uses several controllers", "Each switch is assigned to one controller; the assignment is usually fixed when the network is designed."],
    [fa.FaChartLine, "Traffic changes during the day", "Peaks overload some controllers (slow flow set-up) while quiet periods leave others idle but still using energy."],
    [fa.FaExclamationTriangle, "Existing fixes are partial", "Static placement cannot adapt; threshold rules react only after overload; migration alone cannot save energy."],
  ];
  for (let i = 0; i < rows.length; i++) {
    const y = 1.75 + i * 1.65;
    await iconCircle(s, rows[i][0], 0.6, y, 0.85, HEX.lt2, HEX.accent1, `Reason ${i + 1}`);
    s.addText(rows[i][1], { x: 1.7, y: y - 0.05, w: 6.4, h: 0.45, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: `Reason ${i + 1} title` });
    s.addText(rows[i][2], { x: 1.7, y: y + 0.42, w: 6.4, h: 0.9, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: `Reason ${i + 1} text` });
  }
  // load-gauge illustration: three controllers at one moment
  card(s, 8.6, 2.55, 4.15, 4.25, HEX.lt2, "Load illustration card");
  s.addText("Controller load at one moment", { x: 8.85, y: 2.7, w: 3.7, h: 0.4, fontSize: 14, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "Load illustration title" });
  const baseY = 6.1, top = 3.45, H = baseY - top;
  [[0.92, HEX.accent2, "C1", "overloaded"], [0.18, HEX.accent4, "C2", "idle"], [0.55, HEX.accent1, "C3", "normal"]].forEach(([v, col, lab, st], i) => {
    const x = 8.95 + i * 1.0;
    s.addShape(pres.shapes.RECTANGLE, { x, y: baseY - v * H, w: 0.7, h: v * H, fill: { color: col }, line: { color: col }, objectName: `${lab} bar` });
    s.addText([{ text: lab, options: { bold: true, breakLine: true } }, { text: st, options: { fontSize: 11 } }],
      { x: x - 0.2, y: baseY + 0.05, w: 1.1, h: 0.55, fontSize: 13, color: C.text1, align: "center", margin: 0, isTextBox: true, objectName: `${lab} label` });
  });
  [[0.7, "overload 0.70", HEX.accent2], [0.3, "underload 0.30", HEX.dk2]].forEach(([v, lab, col], i) => {
    const y = baseY - v * H;
    s.addShape(pres.shapes.LINE, { x: 8.8, y, w: 2.95, h: 0, line: { color: col, width: 1.25, dashType: "dash" }, objectName: `Threshold line ${i + 1}` });
    s.addText(lab.replace(" ", "\n"), { x: 11.8, y: y - 0.25, w: 0.9, h: 0.5, fontSize: 11, color: col, align: "left", margin: 0, isTextBox: true, objectName: `Threshold label ${i + 1}` });
  });
  camGuide(s);
  s.addNotes(
`(About 40 seconds.) Software-defined networking separates the control plane from the switches. Large networks use several controllers, and each switch is assigned to one of them. The problem is that traffic is not constant. During a peak, one controller can be overloaded, as C1 is here, which slows down flow set-up. During quiet periods, other controllers sit almost idle, like C2, but they still consume energy. The usual answers only solve part of this. Static controller placement is decided once at design time. Threshold rules react only after a controller is already overloaded. And switch migration can balance load but cannot save energy, because the number of active controllers never changes. So we need a policy that decides at runtime, and ideally before the overload happens.`);

  // ============ 3. Aim & objectives ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  title(s, "Research aim and objectives");
  card(s, 0.6, 1.6, 9.5, 1.15, HEX.lt2, "Aim card");
  s.addText([{ text: "Aim: ", options: { bold: true, color: HEX.accent6 } },
    { text: "design and evaluate a Rainbow DQN agent that proactively manages distributed SDN controllers at runtime, jointly choosing migrate, park, evoke or no action to balance latency, energy and load." }],
    { x: 0.85, y: 1.68, w: 9.1, h: 1.0, fontSize: 16, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: "Aim" });
  const objs = [
    [fa.FaProjectDiagram, "O1  Environment", "Build an SDN controller-management environment on real Topology Zoo graphs and temporal traffic, with valid-action checks and latency, energy and balance metrics."],
    [fa.FaBrain, "O2  Learning and fair evaluation", "Train Rainbow DQN agents on five topologies (9–197 switches) and compare them with threshold, random-valid and no-op baselines on held-out traffic."],
    [fa.FaChartLine, "O3  Forecast-assisted proactivity", "Add a GRU traffic forecaster and test whether predicting load improves decisions, on synthetic and real (Abilene) traffic."],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.13;
    card(s, x, 3.1, 3.83, 3.65, HEX.lt1, `Objective ${i + 1} card`);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.1, w: 3.83, h: 3.65, rectRadius: 0.08, fill: { color: HEX.lt1 },
      line: { color: "D5DEE9", width: 1 }, shadow: { type: "outer", color: "15233B", opacity: 0.12, blur: 6, offset: 2, angle: 90 }, objectName: `Objective ${i + 1} frame` });
    await iconCircle(s, objs[i][0], x + 0.3, 3.35, 0.8, HEX.accent1, HEX.lt1, `Objective ${i + 1}`);
    s.addText(objs[i][1], { x: x + 0.3, y: 4.3, w: 3.3, h: 0.75, fontSize: 17, valign: "top", bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: `Objective ${i + 1} title` });
    s.addText(objs[i][2], { x: x + 0.3, y: 5.1, w: 3.3, h: 1.55, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: `Objective ${i + 1} text` });
  }
  camGuide(s);
  s.addNotes(
`(About 35 seconds.) My aim was to design and evaluate a Rainbow DQN agent that manages the controllers proactively while the network runs. It has four choices at each step: migrate a switch, park a controller, evoke a parked controller, or do nothing. I had three objectives. First, build an environment based on real network topologies and time-ordered traffic, where every action is checked before it is executed. Second, train agents on five topologies of very different sizes and compare them fairly against three baselines on traffic they have never seen. Third, add a GRU forecaster and find out whether predicting future load actually improves the decisions.`);

  // ============ 4. Literature review (1) ============
  pres.addSection({ title: "Literature review" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Literature review" });
  title(s, "Related work: four lines, each solves part of the problem");
  const lit = [
    [fa.FaMapMarkerAlt, "Design-time placement", "Heller 2012 [2]; Das 2020 [3]", "Optimises where controllers go and which switches they serve.", "Decided once, offline: cannot follow daily traffic or save energy."],
    [fa.FaExchangeAlt, "Runtime switch migration", "ElastiCon [4]; SAR-LB (RL) [5]; survey [11]", "Rebalances load by moving switches; RL beats hand-made rules.", "Active controller set fixed: no park/evoke, so no energy saving."],
    [fa.FaBrain, "DRL-based placement", "DDCP [6]; MOOO-RDQN [7]; AP-DQN [8]", "Rainbow DQN handles large discrete SDN action spaces well.", "Still a placement decision, not runtime controller management."],
    [fa.FaChartLine, "Forecasting, safety, scale", "Li 2024 [9]; Dinh 2025 [10]; Jha 2026 [16]", "Forecasts (CNN-LSTM), safety layers and federated RL help DRL.", "Forecast benefit never tested against the same agent without it."],
  ];
  for (let i = 0; i < 4; i++) {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 4.75, y = 1.65 + row * 2.68;
    card(s, x, y, 4.5, 2.45, HEX.lt2, `Lit ${i + 1} card`);
    await iconCircle(s, lit[i][0], x + 0.22, y + 0.22, 0.6, HEX.accent1, HEX.lt1, `Lit ${i + 1}`);
    s.addText(lit[i][1], { x: x + 0.95, y: y + 0.18, w: 3.4, h: 0.38, fontSize: 17, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: `Lit ${i + 1} title` });
    s.addText(lit[i][2], { x: x + 0.95, y: y + 0.55, w: 3.45, h: 0.3, fontSize: 11, italic: true, color: C.text2, margin: 0, isTextBox: true, objectName: `Lit ${i + 1} refs` });
    s.addText([{ text: "+ ", options: { bold: true, color: HEX.accent3 } }, { text: lit[i][3], options: { breakLine: true } },
      { text: "– ", options: { bold: true, color: HEX.accent2 } }, { text: lit[i][4] }],
      { x: x + 0.22, y: y + 1.0, w: 4.1, h: 1.35, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 4, isTextBox: true, objectName: `Lit ${i + 1} text` });
  }
  // right column (below the webcam corner): synthesis
  card(s, 10.35, 2.6, 2.4, 4.2, HEX.dk1, "Synthesis card");
  s.addText([{ text: "Synthesis", options: { bold: true, fontSize: 17, color: HEX.accent4, breakLine: true } },
    { text: "Each line optimises one lever. No study lets a single learned policy pull all of them at runtime.", options: { fontSize: 15, color: HEX.lt1, breakLine: true } },
    { text: "This work: migrate + park + evoke in one runtime policy, with and without a forecast.", options: { fontSize: 15, bold: true, color: "9CC3F0" } }],
    { x: 10.55, y: 2.8, w: 2.05, h: 3.8, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true, objectName: "Synthesis text" });
  camGuide(s);
  s.addNotes(
`(About 45 seconds.) The literature falls into four lines of work. Design-time placement, from Heller and the Das survey, decides where controllers go, but only once. Runtime migration, such as ElastiCon and the reinforcement-learning method SAR-LB, moves switches to rebalance load, but keeps the same controllers switched on, so it cannot save energy. DRL placement work, especially MOOO-RDQN, showed that Rainbow DQN works well on SDN problems, and that is why I chose Rainbow, but it still answers a design-time question. Finally, forecasting, safety and federated studies improve DRL controllers, but none of them checks whether the forecast helps compared with the same agent without it. So each line optimises one lever, and none lets one learned policy use all of them at runtime.`);

  // ============ 5. Literature review (2): gap ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Literature review" });
  title(s, "Critical comparison and the research gap");
  const Y = { text: "✓", options: { color: HEX.accent3, bold: true, align: "center" } };
  const N = { text: "✗", options: { color: HEX.accent2, bold: true, align: "center" } };
  const P = { text: "partly", options: { color: HEX.dk2, align: "center", fontSize: 12 } };
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk1 }, align: "center", valign: "middle" } });
  const nm = (t, b) => ({ text: t, options: { bold: !!b, color: HEX.dk1 } });
  const tbl = [
    [hdr("Approach"), hdr("Runtime decision"), hdr("Migrate"), hdr("Park / evoke (energy)"), hdr("Forecast"), hdr("With vs without forecast")],
    [nm("Placement studies [2, 3, 6, 8]"), N, N, P, N, N],
    [nm("SAR-LB [5]"), Y, Y, N, N, N],
    [nm("MOOO-RDQN [7]"), N, P, N, P, N],
    [nm("Li et al., MEC [9]"), N, N, N, Y, N],
    [nm("This work", true), Y, Y, Y, Y, Y],
  ];
  s.addTable(tbl, { x: 0.6, y: 1.65, w: 9.4, colW: [2.95, 1.2, 1.0, 1.5, 1.1, 1.65], rowH: 0.62, fontSize: 14,
    fontFace: THEME.bodyFontFace, color: HEX.dk1, valign: "middle", border: { type: "solid", pt: 0.75, color: "D5DEE9" },
    fill: { color: HEX.lt1 }, objectName: "Comparison table" });
  s.addText("Partly: MOOO-RDQN monitors traffic and maps switches at design time; placement work such as DDCP can change the number of controllers.",
    { x: 0.6, y: 5.55, w: 9.4, h: 0.55, fontSize: 11, italic: true, color: C.text2, margin: 0, isTextBox: true, objectName: "Table note" });
  card(s, 10.35, 2.6, 2.4, 4.2, HEX.lt2, "Gap card");
  await iconCircle(s, fa.FaBullseye, 10.6, 2.8, 0.65, HEX.accent2, HEX.lt1, "Gap");
  s.addText([{ text: "Research gap", options: { bold: true, fontSize: 17, breakLine: true } },
    { text: "One learned runtime policy that migrates, parks and evokes, tested with and without forecasting on time-ordered, held-out traffic.", options: { fontSize: 14 } }],
    { x: 10.55, y: 3.6, w: 2.05, h: 3.1, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true, objectName: "Gap text" });
  s.addText("Published percentage gains use different problems, traffic and metrics, so the comparison is methodological, not a numerical ranking.",
    { x: 0.6, y: 6.2, w: 9.4, h: 0.6, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, objectName: "Comparison caveat" });
  camGuide(s);
  s.addNotes(
`(About 35 seconds.) This table compares the closest work feature by feature. Placement studies are not runtime decisions. SAR-LB decides at runtime but only migrates. MOOO-RDQN uses the same Rainbow algorithm but for placement. Li et al. use forecasting, but in a multi-access edge computing setting, and never compare against the same agent without the forecast. That leaves the gap my work addresses: a single learned runtime policy that can migrate, park and evoke, evaluated with and without forecasting on held-out, time-ordered traffic. Because these papers use different problems and metrics, I compare them by method, not by quoting their percentages against my numbers.`);

  // ============ 6. Methodology (1) ============
  pres.addSection({ title: "Methodology" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Methodology" });
  title(s, "Proposed approach: Rainbow DQN as the master controller");
  s.addImage({ path: path.join(FIG, "architecture.png"), x: 0.6, y: 1.55, w: 7.35, h: 5.46, altText: "System architecture: data layer, Gymnasium environment, Rainbow DQN agent and optional GRU forecaster", objectName: "Architecture diagram" });
  const meth = [
    ["State", "Controller loads, active/parked flags, switch→controller map (+ GRU forecast in predictive mode)"],
    ["Actions", "No-op · migrate switch i · evoke controller j · park controller j (N + 11 slots), masked to valid ones"],
    ["Reward", "−(0.35 latency + 0.35 energy + 0.30 imbalance) + action terms, clipped to [−2, 1]"],
    ["Agent", "Rainbow: double Q, prioritised replay, dueling, n-step, noisy nets, C51"],
  ];
  for (let i = 0; i < meth.length; i++) {
    const y = 2.5 + i * 1.12;
    card(s, 8.25, y, 4.5, 1.0, HEX.lt2, `Method ${i + 1} card`);
    s.addText([{ text: meth[i][0] + "  ", options: { bold: true, color: HEX.accent6 } }, { text: meth[i][1] }],
      { x: 8.42, y: y + 0.05, w: 4.2, h: 0.9, fontSize: 14, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `Method ${i + 1}` });
  }
  camGuide(s);
  s.addNotes(
`(About 50 seconds.) Here is the architecture. On the left, the data layer turns a Topology Zoo graph into a switch-to-controller delay matrix and supplies a time-ordered traffic trace. In the middle is the environment: one master controller and five slave slots, three active and two parked. It detects overload above 0.70 and underload below 0.30, and builds a mask of valid actions. The Rainbow DQN agent, which runs in the master controller, observes the controller loads, which controllers are active, and the current switch mapping. It chooses one action: do nothing, migrate a switch, evoke a controller or park one. The mask guarantees that every chosen action can actually be executed. The reward combines normalised latency, energy and load imbalance, plus small terms for useful or useless actions. I chose Rainbow because it combines six DQN improvements that suit a large discrete action space with delayed effects. In predictive mode, a GRU forecasts load five steps ahead and adds it to the state, so capacity can be added before overload; parking still uses only current load, so a forecast never removes capacity early.`);

  // ============ 7. Methodology (2) ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Methodology" });
  title(s, "Workflow, data and fair evaluation protocol");
  s.addImage({ path: path.join(FIG, "protocol.png"), x: 0.6, y: 1.6, w: 9.5, h: 3.35, altText: "Experimental workflow from data preparation to single-use final test", objectName: "Protocol diagram" });
  const tiles = [
    ["5", "Topology Zoo networks, 9 to 197 switches"],
    ["70/15/15", "chronological train / validation / test split"],
    ["3", "baselines: threshold, random-valid, no-op"],
    ["3 seeds", "on real Abilene traffic mapped to OS3E"],
    ["Mininet", "OVS + os-ken controllers run the same policies"],
  ];
  for (let i = 0; i < tiles.length; i++) {
    const x = 0.6 + i * 2.45;
    card(s, x, 5.25, 2.25, 1.6, HEX.lt2, `Tile ${i + 1} card`);
    s.addText(tiles[i][0], { x: x + 0.15, y: 5.33, w: 1.95, h: 0.6, fontFace: THEME.headFontFace, fontSize: 26, bold: true, color: HEX.accent1, margin: 0, isTextBox: true, objectName: `Tile ${i + 1} value` });
    s.addText(tiles[i][1], { x: x + 0.15, y: 5.93, w: 1.95, h: 0.85, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `Tile ${i + 1} label` });
  }
  camGuide(s);
  s.addNotes(
`(About 40 seconds.) To make the evaluation credible, I followed a strict protocol. All traces are split in time order, 70, 15 and 15 percent, so the model never sees future traffic. The GRU and the agents are trained on the training split only. All tuning, such as choosing current or predictive mode per topology, used validation data, and each test split was used once, after freezing. Every policy runs on identical windows and seeds. I used five real topologies from 9 to 197 switches, a synthetic trace generator, and real Abilene traffic mapped to OS3E, retrained with three seeds. Finally, I ran the same trained policies on an emulated network in Mininet with Open vSwitch and separate os-ken controllers.`);

  // ============ 8. Results (1): synthetic ============
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  title(s, "Learned policy wins on all five topologies, mainly by saving energy");
  const topo = ["Gridnet", "BellCanada", "OS3E*", "Interoute", "Cogentco"];
  s.addChart(pres.charts.BAR, [{ name: "Reward gain", labels: topo, values: [28.64, 24.27, 55.59, 52.40, 67.07] }], {
    x: 0.6, y: 1.55, w: 4.75, h: 4.7, barDir: "col", chartColors: [HEX.accent1], ...chartBase(),
    title: "Reward gain over threshold (points)", dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.0",
    valAxisMinVal: 0, valAxisMaxVal: 80, valAxisMajorUnit: 20, showLegend: false, objectName: "Reward gain chart" });
  s.addChart(pres.charts.BAR, [{ name: "Energy saving", labels: topo, values: [4.2, 10.8, 14.5, 16.8, 19.6] }], {
    x: 5.5, y: 1.55, w: 4.75, h: 4.7, barDir: "col", chartColors: [HEX.accent3], ...chartBase(),
    title: "Energy saved vs threshold (%)", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    valAxisMinVal: 0, valAxisMaxVal: 25, valAxisMajorUnit: 5, showLegend: false, objectName: "Energy saving chart" });
  s.addText("Held-out synthetic test, 6 × 500 steps. *OS3E uses predictive GRU-RDQN, the others current-state RDQN; Cogentco trained for 120k steps.",
    { x: 0.6, y: 6.3, w: 9.6, h: 0.5, fontSize: 11, italic: true, color: C.text2, margin: 0, isTextBox: true, objectName: "Chart note" });
  stat(s, 10.55, 2.55, 2.3, "5 / 5", "topologies: highest reward vs all baselines", HEX.accent1, "Stat wins");
  stat(s, 10.55, 4.0, 2.3, "0", "failed actions in any final evaluation", HEX.accent3, "Stat failures");
  stat(s, 10.55, 5.45, 2.3, "≤ 0.8 ms", "extra avg latency on 4 of 5 (the trade-off)", HEX.accent2, "Stat tradeoff");
  camGuide(s);
  s.addNotes(
`(About 45 seconds.) These are the main results on held-out synthetic traffic. On all five topologies, the learned policy had the highest reward compared with all three baselines. The left chart shows the reward gain over the threshold rule, from about 24 to 67 points. The right chart shows why: the learned policy used 4 to 20 percent less energy, and the saving grows with network size. It learned to keep fewer controllers on and to add capacity only when it pays off. No action failed in any evaluation, thanks to the valid-action mask. The honest trade-off is that average latency was slightly higher, by 0.2 to 0.8 milliseconds, on four topologies, because parking a controller moves its switches further away. That is the balance the reward was designed to make.`);

  // ============ 9. Results (2): forecasting ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  title(s, "Does forecasting help? It depends on the scenario");
  const verdicts = [
    [HEX.accent3, "Helps", "OS3E (synthetic)", "Reward −478.5 → −452.9, energy 442 → 381 W vs current RDQN; most forecast-triggered actions."],
    [HEX.accent2, "Does not help", "Gridnet, BellCanada", "Good forecasts, but current-state RDQN kept the higher reward (also at horizon 3)."],
    [HEX.accent4, "Inconclusive", "Real Abilene traffic", "Zero-shot transfer works; over 3 seeds the means differ but each mode wins some seeds."],
  ];
  verdicts.forEach(([col, tag, where, txt], i) => {
    const y = 1.6 + i * 1.75;
    card(s, 0.6, y, 5.9, 1.55, HEX.lt2, `Verdict ${i + 1} card`);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: y + 0.2, w: 1.55, h: 0.42, rectRadius: 0.08, fill: { color: col }, line: { color: col }, objectName: `Verdict ${i + 1} tag` });
    s.addText(tag, { x: 0.8, y: y + 0.2, w: 1.55, h: 0.42, fontSize: 13, bold: true, color: HEX.dk1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `Verdict ${i + 1} tag text` });
    s.addText(where, { x: 2.5, y: y + 0.2, w: 3.85, h: 0.42, fontSize: 16, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `Verdict ${i + 1} where` });
    s.addText(txt, { x: 0.8, y: y + 0.7, w: 5.55, h: 0.8, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: `Verdict ${i + 1} text` });
  });
  s.addChart(pres.charts.BAR, [
    { name: "Current RDQN", labels: ["Seed 42", "Seed 43", "Seed 44", "Mean"], values: [-107.39, -113.20, -144.82, -121.80] },
    { name: "Predictive GRU-RDQN", labels: ["Seed 42", "Seed 43", "Seed 44", "Mean"], values: [-122.71, -119.84, -91.02, -111.19] },
  ], { x: 6.8, y: 2.45, w: 5.95, h: 4.4, barDir: "col", chartColors: [HEX.accent1, HEX.accent2], ...chartBase(),
    title: "Real Abilene traffic: reward per seed", dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisMinVal: -160, valAxisMaxVal: 0, valAxisMajorUnit: 40, showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.dk1,
    objectName: "Seed reward chart" });
  camGuide(s);
  s.addNotes(
`(About 45 seconds.) The third objective asked whether forecasting helps, and the honest answer is: it depends. On OS3E with synthetic traffic, the GRU clearly helped: reward improved from minus 478 to minus 453, and energy dropped from 442 to 381 watts, compared with the current-state agent. On Gridnet and BellCanada, the forecasts were accurate, but the current-state agent still had the better reward, even with a shorter horizon. On real Abilene traffic, the chart shows the three training seeds. The predictive model has the better mean, but the current model wins on seeds 42 and 43, and the predictive one wins only on seed 44. With that spread, I do not claim a significant difference. Forecasting can make management more proactive, but its benefit is scenario-dependent.`);

  // ============ 10. Results (3): Mininet ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  title(s, "Validation on an emulated OpenFlow network (Mininet)");
  s.addImage({ path: path.join(FIG, "mininet_architecture.png"), x: 0.6, y: 1.55, w: 6.55, h: 4.41, altText: "Mininet emulation: orchestrator, Open vSwitch data plane and os-ken controller processes", objectName: "Mininet diagram" });
  s.addText("Same OS3E policies, unchanged; 3 windows × 150 steps; 24 policy runs on one host. Energy and control latency remain modelled.",
    { x: 0.6, y: 6.1, w: 6.55, h: 0.7, fontSize: 12, italic: true, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "Mininet note" });
  const mstats = [
    ["687 / 0", "network operations / failures", HEX.accent3],
    ["97.4%", "action agreement with simulator (GRU-RDQN: 100%)", HEX.accent1],
    ["40 ms", "per switch handover; 0.65 s per controller start", HEX.accent6],
    ["3 / 3", "predictive runs won by GRU-RDQN, 18–32% less energy", HEX.accent2],
  ];
  mstats.forEach(([v, l, col], i) => {
    const col2 = i % 2, row = Math.floor(i / 2);
    const x = 7.5 + col2 * 2.7, y = 2.55 + row * 2.2;
    card(s, x, y, 2.5, 2.0, HEX.lt2, `E4 stat ${i + 1} card`);
    s.addText(v, { x: x + 0.15, y: y + 0.15, w: 2.2, h: 0.7, fontFace: THEME.headFontFace, fontSize: 30, bold: true, color: col, margin: 0, isTextBox: true, objectName: `E4 stat ${i + 1} value` });
    s.addText(l, { x: x + 0.15, y: y + 0.88, w: 2.2, h: 1.05, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `E4 stat ${i + 1} label` });
  });
  camGuide(s);
  s.addNotes(
`(About 45 seconds.) To check that the decisions work on a real OpenFlow data plane, I built an emulation in Mininet. It has 34 Open vSwitch switches for OS3E and five separate os-ken controller processes, and it replays the same traffic and measures the switch loads. The trained policies ran unchanged. All 687 controller start, stop and handover operations succeeded. The emulated decisions matched the simulator on 97.4 percent of steps, and the GRU-RDQN matched it exactly. A switch handover took about 40 milliseconds and a controller start 0.65 seconds, far below the 5-second decision interval. GRU-RDQN had the best reward in all three predictive runs, using 18 to 32 percent less energy than the threshold rule. Energy and control latency are still model values here, so this validates executability and cost, not a measured energy saving.`);

  // ============ 11. Conclusion & future work ============
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
  s.addText("Conclusion and future work", { x: 0.6, y: 0.4, w: 9.5, h: 0.95, fontFace: THEME.headFontFace, fontSize: 32, bold: true,
    color: HEX.lt1, valign: "middle", margin: 0, isTextBox: true, objectName: "Title" });
  const concl = [
    "A single Rainbow DQN policy that migrates, parks and evokes beat threshold, random and no-op control on all five topologies, using 4–20% less energy.",
    "The valid-action mask kept every decision executable: 0 failures in simulation and in Mininet.",
    "GRU forecasting helps in some scenarios (OS3E), not others; its value is scenario-dependent.",
    "Contribution: runtime formulation, leakage-free evaluation protocol, forecast ablation and a Mininet backend.",
  ];
  s.addText("Key findings", { x: 0.6, y: 2.55, w: 5.9, h: 0.45, fontSize: 20, bold: true, color: HEX.accent4, margin: 0, isTextBox: true, objectName: "Findings heading" });
  s.addText(concl.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < concl.length - 1 } })),
    { x: 0.6, y: 3.15, w: 5.9, h: 3.5, fontSize: 15, color: HEX.lt1, margin: 0, valign: "top", paraSpaceAfter: 10, isTextBox: true, objectName: "Findings" });
  const fut = [
    [fa.FaServer, "Multi-host testbed with ONOS / OpenDaylight and measured power"],
    [fa.FaLayerGroup, "More seeds, paired tests, reward and threshold sensitivity"],
    [fa.FaProjectDiagram, "Graph-based policies that transfer across topologies"],
    [fa.FaShieldAlt, "Safety layers and topology-aware forecasting"],
  ];
  s.addText("Future work", { x: 7.0, y: 2.55, w: 5.8, h: 0.45, fontSize: 20, bold: true, color: HEX.accent4, margin: 0, isTextBox: true, objectName: "Future heading" });
  for (let i = 0; i < fut.length; i++) {
    const y = 3.15 + i * 0.95;
    await iconCircle(s, fut[i][0], 7.0, y, 0.62, HEX.accent1, HEX.lt1, `Future ${i + 1}`);
    s.addText(fut[i][1], { x: 7.8, y: y - 0.05, w: 4.95, h: 0.72, fontSize: 15, color: HEX.lt1, margin: 0, valign: "middle", isTextBox: true, objectName: `Future ${i + 1} text` });
  }
  s.addText("Thank you", { x: 0.6, y: 6.75, w: 4, h: 0.45, fontFace: THEME.headFontFace, fontSize: 18, italic: true, color: "C9D6E8", margin: 0, isTextBox: true, objectName: "Thanks" });
  camGuide(s);
  s.addNotes(
`(About 40 seconds.) To conclude. A single Rainbow DQN policy that can migrate, park and evoke controllers beat the threshold, random and no-op policies on all five topologies, mainly by using 4 to 20 percent less energy. The valid-action mask made every decision executable, with zero failures in both simulation and Mininet. Forecasting helps in some scenarios but not all, so its value is scenario-dependent. My contributions are the runtime formulation, a leakage-free evaluation protocol, a matched forecast ablation, and the emulation backend. In future, I would test on multiple hosts with ONOS or OpenDaylight and measured power, use more seeds and sensitivity analysis, and build graph-based policies that transfer between topologies. Thank you.`);

  // ============ 12. References ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  title(s, "References");
  const refs = [
    "N. McKeown et al., “OpenFlow: Enabling innovation in campus networks,” ACM SIGCOMM CCR, vol. 38, no. 2, 2008.",
    "B. Heller, R. Sherwood, N. McKeown, “The controller placement problem,” HotSDN, 2012.",
    "T. Das, V. Sridharan, M. Gurusamy, “A survey on controller placement in SDN,” IEEE COMST, vol. 22, no. 1, 2020.",
    "A. Dixit et al., “Towards an elastic distributed SDN controller,” ACM SIGCOMM CCR, vol. 43, no. 4, 2013.",
    "S. Yeo et al., “Achieving balanced load distribution with RL-based switch migration in distributed SDN controllers,” Electronics, vol. 10, 2021.",
    "E. H. Bouzidi et al., “Dynamic clustering of SDN switches and controller placement using DRL,” Computer Networks, vol. 207, 2022.",
    "J. Chen et al., “MOOO-RDQN: A DRL based method for multi-objective optimization of controller placement and traffic monitoring in SDN,” JNCA, vol. 242, 2025.",
    "R. Farahi et al., “AP-DQN: A novel approach for controller placement in SDN using DRL,” Results in Engineering, vol. 29, 2026.",
    "C. Li et al., “DRL based controller placement and optimal edge selection in SDN-based MEC environments,” JPDC, vol. 193, 2024.",
    "L. Dinh, P. T. A. Quang, J. Leguay, “Safe load balancing in software-defined-networking,” Computer Communications, vol. 229, 2025.",
    "Y. Liu et al., “Controller load optimization strategies in SDN: A survey,” JNCA, vol. 233, 2025.",
    "M. Hessel et al., “Rainbow: Combining improvements in deep reinforcement learning,” AAAI, 2018.",
    "K. Cho et al., “Learning phrase representations using RNN encoder–decoder,” EMNLP, 2014.",
    "S. Knight et al., “The Internet Topology Zoo,” IEEE JSAC, vol. 29, no. 9, 2011.",
    "B. Lantz, B. Heller, N. McKeown, “A network in a laptop: Rapid prototyping for SDN,” HotNets, 2010.",
    "A. K. Jha et al., “Asynchronous federated RL for scalable load balancing in SDN,” Discover Computing, vol. 29, 2026.",
  ];
  const refRuns = (list, start) => list.map((r, i) => ({ text: `[${start + i}]  ${r}`, options: { breakLine: i < list.length - 1 } }));
  s.addText(refRuns(refs.slice(0, 8), 1), { x: 0.6, y: 1.55, w: 6.0, h: 5.3, fontSize: 11, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true, objectName: "References left" });
  s.addText(refRuns(refs.slice(8), 9), { x: 6.85, y: 2.5, w: 5.9, h: 4.35, fontSize: 11, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true, objectName: "References right" });
  camGuide(s);
  s.addNotes(`(About 10 seconds; do not read out.) These are the main references cited in the slides. The full IEEE reference list is in the report.`);

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("Wrote " + OUT);
})().catch((e) => { console.error(e); process.exit(1); });
