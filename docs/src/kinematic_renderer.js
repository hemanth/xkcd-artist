/**
 * Procedural Hand-Wobbled Monochrome Stick-Figure, Stage-Prop, Chart, and Flowchart Renderer
 * Driven by `gnumanth/xkcd-artist` ONNX 12-DOF pose kinematics, prop predictions, and neural stroke wobble.
 */

export function renderKinematicComic(spec, helpers) {
  const {
    escapeXml,
    wrapTextPx,
    wobbledCurvePath,
    wobbledPolylinePath,
    wobbledRectPath,
    arrowheadPaths,
    fmt,
    wobbledCurvePoints,
  } = helpers;

  const layout = spec.layout || 'three_panel';
  const prop = spec.prop || 'none';
  const cast = spec.cast?.length ? spec.cast : ['cueball', 'megan'];
  const turns = spec.turns || [];
  const poses12 = spec.neural?.poses_12dof || null;

  if (layout === 'annotated_chart') {
    return renderAnnotatedChart(spec, helpers);
  }
  if (layout === 'flowchart_diagram') {
    return renderFlowchartDiagram(spec, helpers);
  }

  function wobbledOvalPath(cx, cy, rx, ry, { width = 2.05, wobble = 0.45, fill = '#ffffff', seed = 1.0 } = {}) {
    const wScale = helpers.wobbleScale || 1.0;
    const sScale = helpers.strokeScale || 1.0;
    const steps = 28;
    const pts = [];
    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * Math.PI * 2;
      const jitter =
        wobble *
        wScale *
        (0.6 * Math.sin(t * 3 + seed * 1.9) + 0.4 * Math.cos(t * 5 - seed * 2.7));
      const x = cx + (rx + jitter) * Math.cos(t);
      const y = cy + (ry + jitter * 0.85) * Math.sin(t);
      pts.push(`${i === 0 ? 'M' : 'L'} ${fmt(x)} ${fmt(y)}`);
    }
    return `<path d="${pts.join(' ')} Z" fill="${fill}" stroke="#0a0a0a" stroke-width="${Number((width * sScale).toFixed(2))}" stroke-linecap="round" stroke-linejoin="round"/>`;
  }

  function drawCharacterHead(hx, hy, character, facing, seed) {
    const rx = 12.5;
    const ry = 14.0;
    const parts = [wobbledOvalPath(hx, hy, rx, ry, { width: 2.05, wobble: 0.42, fill: '#ffffff', seed })];
    let crownY = hy - ry - 3;

    if (character === 'megan') {
      // Classic XKCD Megan silhouette: solid dark top cap crescent + sleek side bob curtains
      const capPath = [
        `M ${fmt(hx - 12.6)} ${fmt(hy - 4.5)}`,
        `Q ${fmt(hx)} ${fmt(hy - 20.5)} ${fmt(hx + 12.6)} ${fmt(hy - 4.5)}`,
        `Q ${fmt(hx)} ${fmt(hy - 12.2)} ${fmt(hx - 12.6)} ${fmt(hy - 4.5)} Z`,
      ].join(' ');
      parts.push(`<path d="${capPath}" fill="#0a0a0a" stroke="#0a0a0a" stroke-width="1.6" stroke-linejoin="round"/>`);
      parts.push(
        wobbledCurvePath(hx - 12.5, hy - 5, hx - 13.4, hy + 11.5, {
          width: 2.5,
          curveX: -1.8,
          wobble: 0.32,
          seed: seed + 1.1,
        })
      );
      parts.push(
        wobbledCurvePath(hx + 12.5, hy - 5, hx + 13.4, hy + 11.5, {
          width: 2.5,
          curveX: 1.8,
          wobble: 0.32,
          seed: seed + 1.5,
        })
      );
      crownY = hy - ry - 4;
    } else if (character === 'ponytail') {
      const bx = hx - facing * 12.5;
      const by = hy - 2;
      parts.push(wobbledOvalPath(bx - facing * 1.5, by, 2.4, 2.8, { width: 1.8, wobble: 0.2, fill: '#0a0a0a', seed }));
      parts.push(
        wobbledCurvePath(bx - facing * 2, by - 1, bx - facing * 12, by + 13, {
          width: 2.1,
          curveX: -facing * 4,
          curveY: -4,
          wobble: 0.4,
          seed: seed + 1.7,
        })
      );
      parts.push(
        wobbledCurvePath(bx - facing * 2, by + 1, bx - facing * 9, by + 16, {
          width: 1.9,
          curveX: -facing * 2,
          curveY: -2,
          wobble: 0.4,
          seed: seed + 2.3,
        })
      );
    } else if (character === 'black_hat') {
      const brimY = hy - 10.5;
      const hatTop = brimY - 18;
      const crownPts = [
        [hx - 10, brimY],
        [hx - 11, hatTop],
        [hx + 11, hatTop],
        [hx + 10, brimY],
      ];
      const poly = crownPts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'} ${fmt(x)} ${fmt(y)}`).join(' ') + ' Z';
      parts.push(`<path d="${poly}" fill="#0a0a0a" stroke="#0a0a0a" stroke-width="1.8" stroke-linejoin="round"/>`);
      parts.push(wobbledCurvePath(hx - 17.5, brimY, hx + 17.5, brimY, { width: 2.4, wobble: 0.3, seed: seed + 0.9 }));
      crownY = hatTop - 4;
    } else if (character === 'beret_guy') {
      const by = hy - 10;
      const d = [
        `M ${fmt(hx - 13)} ${fmt(by)}`,
        `Q ${fmt(hx - facing * 5)} ${fmt(by - 15)} ${fmt(hx + 13)} ${fmt(by - 1)}`,
        `Q ${fmt(hx)} ${fmt(by + 2)} ${fmt(hx - 13)} ${fmt(by)} Z`,
      ].join(' ');
      parts.push(`<path d="${d}" fill="#0a0a0a" stroke="#0a0a0a" stroke-width="1.6" stroke-linejoin="round"/>`);
      parts.push(wobbledCurvePath(hx - facing * 2, by - 11, hx - facing * 4, by - 15, { width: 1.9, wobble: 0.2, seed }));
      crownY = by - 16;
    } else if (character === 'hairbun') {
      const bx = hx - facing * 11.5;
      const by = hy - 10.5;
      parts.push(wobbledOvalPath(bx, by, 5.5, 5.5, { width: 1.9, wobble: 0.3, fill: '#ffffff', seed: seed + 0.6 }));
      parts.push(wobbledCurvePath(hx - 9, hy - 11, hx + 8, hy - 9, { width: 1.7, curveY: -2, wobble: 0.25, seed }));
      crownY = Math.min(crownY, by - 7);
    }

    return { svg: parts.join('\n    '), crownX: hx, crownY };
  }

  function drawStickFigure({
    cx,
    floorY,
    character = 'cueball',
    pose = 'conversing',
    facing = 1,
    seated = false,
    tilt = 0,
    armBias = 0,
    seed = 1.0,
  }) {
    const els = [];
    const hipX = cx;
    const hipY = seated === 'couch' ? floorY - 26 : seated ? floorY - 30 : floorY - 45;
    const neckX = cx + facing * (seated ? 2.5 : 1.2) + tilt * 3.5;
    const neckY = hipY - (seated ? 39 : 42);
    const headX = neckX + tilt * 2.0;
    const headY = neckY - 14.0;
    const shoulderX = neckX - facing * 0.4;
    const shoulderY = neckY + 6.5;

    // Office chair if seated at desk
    if (seated === 'chair') {
      const backX = hipX - facing * 11;
      els.push(wobbledCurvePath(backX, hipY - 24, backX + facing * 2, hipY + 4, { width: 1.9, curveX: -facing * 2, seed: seed + 5.1 }));
      els.push(wobbledCurvePath(backX + facing * 1, hipY + 4, hipX + facing * 15, hipY + 4, { width: 2.0, seed: seed + 5.4 }));
      els.push(wobbledCurvePath(hipX + facing * 2, hipY + 4, hipX + facing * 2, floorY - 4, { width: 1.9, seed: seed + 5.7 }));
      els.push(wobbledCurvePath(hipX - 11, floorY, hipX + 14, floorY, { width: 1.9, curveY: -3, seed: seed + 6.0 }));
    }

    // Spine (starts right at `neckY`, clipped cleanly by the opaque `#ffffff` head oval)
    els.push(
      wobbledCurvePath(neckX, neckY, hipX, hipY, {
        width: 2.05,
        curveX: -facing * 1.6,
        wobble: 0.55,
        seed: seed + 0.3,
      })
    );

    // Legs
    if (seated) {
      const kneeX = hipX + facing * 19;
      const kneeY = hipY + 2;
      els.push(wobbledCurvePath(hipX, hipY, kneeX, kneeY, { width: 2.0, wobble: 0.45, seed: seed + 1.1 }));
      els.push(wobbledCurvePath(kneeX, kneeY, kneeX + facing * 2, floorY, { width: 2.0, wobble: 0.45, seed: seed + 1.4 }));
      els.push(wobbledCurvePath(hipX, hipY + 1, kneeX - facing * 4, kneeY + 3, { width: 1.9, wobble: 0.45, seed: seed + 1.7 }));
      els.push(wobbledCurvePath(kneeX - facing * 4, kneeY + 3, kneeX - facing * 2, floorY, { width: 1.9, wobble: 0.45, seed: seed + 1.9 }));
    } else {
      els.push(
        wobbledCurvePath(hipX, hipY, cx - facing * 10.5, floorY, {
          width: 2.05,
          curveX: -facing * 1.2,
          wobble: 0.5,
          seed: seed + 1.2,
        })
      );
      els.push(
        wobbledCurvePath(hipX, hipY, cx + facing * 11.5, floorY, {
          width: 2.05,
          curveX: facing * 1.4,
          wobble: 0.5,
          seed: seed + 1.6,
        })
      );
    }

    // Arms (2-segment expressive kinematics)
    const biasY = armBias * 4.0;
    if (pose === 'typing') {
      const e1x = shoulderX + facing * 9;
      const e1y = shoulderY + 15;
      const h1x = shoulderX + facing * 25;
      const h1y = shoulderY + 17;
      els.push(wobbledCurvePath(shoulderX, shoulderY, e1x, e1y, { width: 2.0, wobble: 0.4, seed: seed + 2.1 }));
      els.push(wobbledCurvePath(e1x, e1y, h1x, h1y, { width: 2.0, wobble: 0.4, seed: seed + 2.4 }));
      els.push(wobbledCurvePath(shoulderX, shoulderY + 2, e1x - facing * 2, e1y + 3, { width: 1.9, wobble: 0.4, seed: seed + 2.7 }));
      els.push(wobbledCurvePath(e1x - facing * 2, e1y + 3, h1x - facing * 3, h1y + 2, { width: 1.9, wobble: 0.4, seed: seed + 2.9 }));
    } else if (pose === 'pointing' || pose === 'pointer_rod') {
      // Back arm relaxed
      els.push(wobbledCurvePath(shoulderX, shoulderY, shoulderX - facing * 9, shoulderY + 22, { width: 2.0, curveX: -facing * 2, seed: seed + 2.1 }));
      // Front arm raised pointing forward
      const ex = shoulderX + facing * 13;
      const ey = shoulderY + 5 + biasY * 0.4;
      const hx2 = shoulderX + facing * 27;
      const hy2 = shoulderY - 5 + biasY * 0.5;
      els.push(wobbledCurvePath(shoulderX, shoulderY, ex, ey, { width: 2.0, wobble: 0.4, seed: seed + 2.4 }));
      els.push(wobbledCurvePath(ex, ey, hx2, hy2, { width: 2.0, wobble: 0.4, seed: seed + 2.7 }));
      if (pose === 'pointer_rod') {
        els.push(wobbledCurvePath(hx2, hy2, hx2 + facing * 22, hy2 - 16, { width: 1.7, wobble: 0.25, seed: seed + 3.0 }));
      }
    } else if (pose === 'chin_think') {
      els.push(wobbledCurvePath(shoulderX, shoulderY, shoulderX - facing * 10, shoulderY + 15, { width: 2.0, seed: seed + 2.1 }));
      els.push(wobbledCurvePath(shoulderX - facing * 10, shoulderY + 15, hipX - facing * 3, hipY - 3, { width: 2.0, seed: seed + 2.3 }));
      const ex = shoulderX + facing * 13;
      const ey = shoulderY + 13;
      els.push(wobbledCurvePath(shoulderX, shoulderY, ex, ey, { width: 2.0, seed: seed + 2.6 }));
      els.push(wobbledCurvePath(ex, ey, headX + facing * 7, headY + 12, { width: 2.0, seed: seed + 2.9 }));
    } else if (pose === 'shrug') {
      els.push(wobbledCurvePath(shoulderX, shoulderY, shoulderX - 11, shoulderY + 13, { width: 2.0, seed: seed + 2.1 }));
      els.push(wobbledCurvePath(shoulderX - 11, shoulderY + 13, shoulderX - 21, shoulderY + 2 + biasY * 0.4, { width: 2.0, seed: seed + 2.3 }));
      els.push(wobbledCurvePath(shoulderX, shoulderY, shoulderX + 11, shoulderY + 13, { width: 2.0, seed: seed + 2.6 }));
      els.push(wobbledCurvePath(shoulderX + 11, shoulderY + 13, shoulderX + 21, shoulderY + 2 + biasY * 0.4, { width: 2.0, seed: seed + 2.9 }));
    } else if (pose === 'arms_crossed') {
      els.push(wobbledCurvePath(shoulderX, shoulderY, shoulderX - 10, shoulderY + 15, { width: 2.0, seed: seed + 2.1 }));
      els.push(wobbledCurvePath(shoulderX - 10, shoulderY + 15, shoulderX + 8, shoulderY + 14, { width: 2.0, seed: seed + 2.3 }));
      els.push(wobbledCurvePath(shoulderX, shoulderY, shoulderX + 10, shoulderY + 15, { width: 2.0, seed: seed + 2.6 }));
      els.push(wobbledCurvePath(shoulderX + 10, shoulderY + 15, shoulderX - 7, shoulderY + 16, { width: 2.0, seed: seed + 2.9 }));
    } else if (pose === 'hands_on_head') {
      els.push(wobbledCurvePath(shoulderX, shoulderY, headX - 16, headY + 2, { width: 2.0, seed: seed + 2.1 }));
      els.push(wobbledCurvePath(headX - 16, headY + 2, headX - 10, headY - 5, { width: 2.0, seed: seed + 2.3 }));
      els.push(wobbledCurvePath(shoulderX, shoulderY, headX + 16, headY + 2, { width: 2.0, seed: seed + 2.6 }));
      els.push(wobbledCurvePath(headX + 16, headY + 2, headX + 10, headY - 5, { width: 2.0, seed: seed + 2.9 }));
    } else {
      // Conversing / natural gesture
      els.push(wobbledCurvePath(shoulderX, shoulderY, shoulderX - facing * 9, shoulderY + 23, { width: 2.0, curveX: -facing * 1.5, seed: seed + 2.1 }));
      const ex = shoulderX + facing * 9;
      const ey = shoulderY + 15;
      els.push(wobbledCurvePath(shoulderX, shoulderY, ex, ey, { width: 2.0, seed: seed + 2.5 }));
      els.push(wobbledCurvePath(ex, ey, shoulderX + facing * 20, shoulderY + 8 + biasY * 0.5, { width: 2.0, seed: seed + 2.8 }));
    }

    const head = drawCharacterHead(headX, headY, character, facing, seed + 3.1);
    els.push(head.svg);

    return {
      svg: els.join('\n    '),
      crownX: head.crownX,
      crownY: head.crownY,
    };
  }

  function drawStageProp(propType, px, floorY, seed = 1.0) {
    const els = [];
    if (propType === 'laptop_desk') {
      const deskTop = floorY - 33;
      const deskL = px - 26;
      const deskR = px + 28;
      els.push(wobbledCurvePath(deskL, deskTop, deskR, deskTop, { width: 2.15, wobble: 0.35, seed }));
      els.push(wobbledCurvePath(deskL + 5, deskTop, deskL + 5, floorY, { width: 2.0, wobble: 0.4, seed: seed + 0.4 }));
      els.push(wobbledCurvePath(deskR - 5, deskTop, deskR - 5, floorY, { width: 2.0, wobble: 0.4, seed: seed + 0.8 }));
      // Laptop keyboard base + open lid
      els.push(wobbledCurvePath(px - 14, deskTop - 2, px + 4, deskTop - 2, { width: 2.2, wobble: 0.2, seed: seed + 1.1 }));
      els.push(wobbledCurvePath(px + 4, deskTop - 2, px + 11, deskTop - 20, { width: 2.2, wobble: 0.25, seed: seed + 1.5 }));
    } else if (propType === 'whiteboard') {
      const wbL = px - 32;
      const wbR = px + 32;
      const wbT = floorY - 102;
      const wbB = floorY - 48;
      els.push(wobbledRectPath(wbL, wbT, wbR, wbB, { width: 2.0, wobble: 0.4, fill: '#ffffff' }));
      els.push(wobbledCurvePath(wbL - 3, wbB + 1, wbR + 3, wbB + 1, { width: 2.2, wobble: 0.25, seed: seed + 0.5 }));
      // Mini diagram squiggles on whiteboard
      els.push(wobbledRectPath(wbL + 8, wbT + 9, wbL + 26, wbT + 22, { width: 1.4, wobble: 0.25, fill: '#ffffff' }));
      els.push(wobbledCurvePath(wbL + 26, wbT + 15, wbL + 38, wbT + 15, { width: 1.4, wobble: 0.2, seed: seed + 1.2 }));
      els.push(wobbledRectPath(wbL + 38, wbT + 9, wbR - 8, wbT + 22, { width: 1.4, wobble: 0.25, fill: '#ffffff' }));
      els.push(wobbledCurvePath(wbL + 10, wbT + 38, wbR - 10, wbT + 30, { width: 1.5, curveY: -6, wobble: 0.4, seed: seed + 1.8 }));
    } else if (propType === 'server_rack') {
      const rL = px - 21;
      const rR = px + 21;
      const rT = floorY - 98;
      els.push(wobbledRectPath(rL, rT, rR, floorY, { width: 2.05, wobble: 0.4, fill: '#ffffff' }));
      for (let i = 0; i < 5; i++) {
        const by = rT + 12 + i * 17;
        els.push(wobbledCurvePath(rL + 4, by, rR - 4, by, { width: 1.5, wobble: 0.25, seed: seed + i }));
        els.push(`<circle cx="${rL + 9}" cy="${by - 5}" r="1.7" fill="#0a0a0a"/>`);
        els.push(`<circle cx="${rL + 15}" cy="${by - 5}" r="1.4" fill="#0a0a0a"/>`);
      }
    } else if (propType === 'couch') {
      const cL = px - 28;
      const cR = px + 26;
      const seatY = floorY - 18;
      els.push(wobbledRectPath(cL, floorY - 44, cL + 11, floorY - 3, { width: 1.95, wobble: 0.4, fill: '#ffffff' }));
      els.push(wobbledRectPath(cL + 9, seatY - 8, cR, floorY - 3, { width: 1.95, wobble: 0.4, fill: '#ffffff' }));
      els.push(wobbledRectPath(cR - 9, floorY - 32, cR + 3, floorY - 3, { width: 1.95, wobble: 0.35, fill: '#ffffff' }));
      els.push(wobbledCurvePath(cL + 4, floorY - 3, cL + 3, floorY, { width: 2.0, seed }));
      els.push(wobbledCurvePath(cR - 2, floorY - 3, cR - 1, floorY, { width: 2.0, seed: seed + 0.5 }));
    } else if (propType === 'pedestal') {
      const pL = px - 15;
      const pR = px + 15;
      const pT = floorY - 50;
      els.push(wobbledRectPath(pL - 3, pT - 4, pR + 3, pT, { width: 1.9, wobble: 0.3, fill: '#ffffff' }));
      els.push(wobbledRectPath(pL, pT, pR, floorY, { width: 1.95, wobble: 0.4, fill: '#ffffff' }));
      // Specimen jar / gadget dome on pedestal
      els.push(wobbledRectPath(px - 9, pT - 24, px + 9, pT - 4, { width: 1.8, wobble: 0.35, fill: '#ffffff' }));
      els.push(wobbledCurvePath(px - 11, pT - 24, px + 11, pT - 24, { width: 2.0, wobble: 0.25, seed: seed + 0.7 }));
      els.push(wobbledOvalPath(px, pT - 13, 4.2, 4.5, { width: 1.5, wobble: 0.4, fill: 'none', seed: seed + 1.4 }));
    } else if (propType === 'pointer_stick') {
      const bL = px - 24;
      const bR = px + 24;
      const bT = floorY - 96;
      const bB = floorY - 46;
      els.push(wobbledCurvePath(px - 14, bB, px - 22, floorY, { width: 1.8, wobble: 0.35, seed }));
      els.push(wobbledCurvePath(px + 14, bB, px + 22, floorY, { width: 1.8, wobble: 0.35, seed: seed + 0.4 }));
      els.push(wobbledRectPath(bL, bT, bR, bB, { width: 2.0, wobble: 0.4, fill: '#ffffff' }));
      els.push(
        wobbledPolylinePath(
          [
            [bL + 7, bB - 10],
            [bL + 19, bB - 22],
            [bL + 30, bB - 18],
            [bR - 7, bT + 10],
          ],
          { width: 1.7, wobble: 0.35 }
        )
      );
    }
    return els.join('\n    ');
  }

  // Determine panel array
  const panelCount =
    layout === 'single_panel' ? 1 : layout === 'two_panel' ? 2 : layout === 'three_panel' ? 3 : 4;

  const panelW =
    panelCount === 1 ? 436 : panelCount === 2 ? 262 : panelCount === 3 ? 236 : 198;
  const panelH = panelCount === 1 ? 314 : 294;
  const gap = panelCount === 4 ? 8 : 10;
  const pad = 4;
  const totalW = panelCount * panelW + (panelCount - 1) * gap + pad * 2;
  const totalH = panelH + pad * 2;
  const floorY = panelH - 20;

  // Pose choreography per panel beat
  const defaultPosesLeft = ['conversing', 'pointing', 'shrug', 'conversing'];
  const defaultPosesRight = ['chin_think', 'conversing', 'arms_crossed', 'hands_on_head'];

  const panelSvgs = [];

  for (let pIdx = 0; pIdx < panelCount; pIdx++) {
    const px0 = pad + pIdx * (panelW + gap);
    const py0 = pad;
    const inner = [];

    // Outer wobbled panel frame + subtle floor line
    inner.push(wobbledRectPath(0, 0, panelW, panelH, { width: 2.15, wobble: 0.45, fill: '#ffffff' }));
    inner.push(
      wobbledCurvePath(14, floorY + 0.5, panelW - 14, floorY + 0.5, {
        width: 1.25,
        wobble: 0.35,
        seed: pIdx * 2.1 + 0.5,
      })
    );

    const charL = cast[0] || 'cueball';
    const charR = cast[1] || (cast.length === 1 ? null : 'megan');
    const charThird = panelCount === 1 && cast[2] ? cast[2] : null;

    // Modulate joint tilt & arm bias from ONNX 12-DOF pose output (`[1, 12]`)
    const c1Tilt = poses12 ? Math.max(-0.4, Math.min(0.4, (poses12[2] || 0) * 0.6)) : 0;
    const c1Arm = poses12 ? Math.max(-1.0, Math.min(1.0, poses12[4] || 0)) : 0;
    const c2Tilt = poses12 ? Math.max(-0.4, Math.min(0.4, (poses12[8] || 0) * 0.6)) : 0;
    const c2Arm = poses12 ? Math.max(-1.0, Math.min(1.0, poses12[10] || 0)) : 0;

    // Position stage prop and characters cleanly per panel width
    let leftX = Math.round(panelW * 0.27);
    let rightX = Math.round(panelW * 0.74);
    let leftSeated = false;
    let leftPose = spec.poses?.[pIdx]?.[0] || defaultPosesLeft[pIdx % defaultPosesLeft.length];
    let rightPose = spec.poses?.[pIdx]?.[1] || defaultPosesRight[pIdx % defaultPosesRight.length];

    if (prop === 'laptop_desk') {
      const deskX = charR ? Math.round(panelW * 0.42) : Math.round(panelW * 0.56);
      leftX = deskX - 24;
      leftSeated = 'chair';
      if (!spec.poses?.[pIdx]?.[0]) leftPose = pIdx === panelCount - 1 ? 'shrug' : 'typing';
      if (charR) rightX = Math.min(panelW - 34, deskX + 54);
      inner.push(drawStageProp('laptop_desk', deskX, floorY, pIdx + 1.2));
    } else if (prop === 'couch') {
      const couchX = Math.round(panelW * 0.32);
      leftX = couchX - 4;
      leftSeated = 'couch';
      inner.push(drawStageProp('couch', couchX, floorY, pIdx + 1.5));
      if (charR) rightX = Math.round(panelW * 0.76);
    } else if (prop === 'whiteboard') {
      const wbX = charThird ? 150 : Math.round(panelW * 0.50);
      leftX = Math.max(32, Math.round(panelW * 0.21));
      rightX = Math.min(panelW - 32, Math.round(panelW * 0.80));
      if (!spec.poses?.[pIdx]?.[0] && pIdx === 0) leftPose = 'pointing';
      inner.push(drawStageProp('whiteboard', wbX, floorY, pIdx + 1.8));
    } else if (prop === 'server_rack') {
      const rackX = charR ? Math.round(panelW * 0.82) : Math.round(panelW * 0.68);
      leftX = Math.round(panelW * 0.24);
      if (charR) rightX = Math.round(panelW * 0.54);
      inner.push(drawStageProp('server_rack', rackX, floorY, pIdx + 2.1));
    } else if (prop === 'pedestal') {
      const pedX = Math.round(panelW * 0.50);
      leftX = Math.max(34, Math.round(panelW * 0.23));
      rightX = Math.min(panelW - 34, Math.round(panelW * 0.77));
      if (!spec.poses?.[pIdx]?.[0] && pIdx === 0) leftPose = 'pointing';
      inner.push(drawStageProp('pedestal', pedX, floorY, pIdx + 2.4));
    } else if (prop === 'pointer_stick') {
      const boardX = Math.round(panelW * 0.68);
      leftX = Math.round(panelW * 0.28);
      leftPose = 'pointer_rod';
      if (charR) rightX = Math.min(panelW - 28, Math.round(panelW * 0.86));
      inner.push(drawStageProp('pointer_stick', boardX, floorY, pIdx + 2.7));
    } else if (!charR) {
      leftX = Math.round(panelW * 0.48);
    }

    if (charThird) {
      leftX = 72;
      rightX = 232;
    }

    const figL = drawStickFigure({
      cx: leftX,
      floorY,
      character: charL,
      pose: leftPose,
      facing: 1,
      seated: leftSeated,
      tilt: c1Tilt,
      armBias: c1Arm,
      seed: pIdx * 3.7 + 1.1,
    });
    inner.push(figL.svg);

    let figR = null;
    if (charR) {
      figR = drawStickFigure({
        cx: rightX,
        floorY,
        character: charR,
        pose: rightPose,
        facing: -1,
        seated: false,
        tilt: c2Tilt,
        armBias: c2Arm,
        seed: pIdx * 3.7 + 2.4,
      });
      inner.push(figR.svg);
    }

    let figThird = null;
    if (charThird) {
      figThird = drawStickFigure({
        cx: 356,
        floorY,
        character: charThird,
        pose: 'arms_crossed',
        facing: -1,
        seated: false,
        tilt: 0,
        armBias: 0,
        seed: 9.2,
      });
      inner.push(figThird.svg);
    }

    // Render crisp upper-panel `xkcd-script` dialogue & non-overlapping curved leader lines
    if (panelCount === 1) {
      let curY = 24;
      const singleTurns = turns.slice(0, 3);
      const placedTurns = [];
      singleTurns.forEach((t, tIdx) => {
        const spk = t.speaker === 'right' && figR ? 'right' : t.speaker === 'third' && figThird ? 'third' : 'left';
        const anchorFig = spk === 'right' ? figR : spk === 'third' ? figThird : figL;
        const maxW = charThird ? 176 : singleTurns.length >= 3 ? 228 : 272;
        const fSize = 14.5;
        const lines = wrapTextPx(t.text, maxW, fSize).slice(0, 3);
        const halfW = Math.round(maxW / 2) + 12;
        let rawTx;
        if (charThird) {
          rawTx = spk === 'left' ? 104 : spk === 'right' ? 224 : 338;
        } else {
          rawTx = spk === 'left' ? 132 : panelW - 132;
        }
        const tx = Math.max(halfW, Math.min(panelW - halfW, rawTx));

        const topY = curY;
        lines.forEach((ln, li) => {
          inner.push(
            `<text x="${tx}" y="${curY + li * 16.5}" text-anchor="middle" font-size="${fSize}" fill="#0a0a0a">${escapeXml(ln)}</text>`
          );
        });
        const tBot = curY + (lines.length - 1) * 16.5 + 5;
        placedTurns.push({ spk, anchorFig, tx, topY, tBot, tIdx });
        curY = tBot + 16;
      });

      // Draw chained bubble connectors when the same speaker speaks twice, otherwise direct non-crossing leaders
      placedTurns.forEach((pt, idx) => {
        const laterSame = placedTurns.slice(idx + 1).find(other => other.spk === pt.spk);
        if (laterSame) {
          const bridgeX = pt.spk === 'left' ? pt.tx - 24 : pt.tx + 24;
          inner.push(
            wobbledCurvePath(bridgeX, pt.tBot, bridgeX, laterSame.topY - 14, {
              width: 1.85,
              curveX: pt.spk === 'left' ? -5 : 5,
              wobble: 0.35,
              seed: pt.tIdx + 1.3,
            })
          );
        } else if (pt.tBot < pt.anchorFig.crownY - 4) {
          const leadStartX =
            charThird && pt.spk === 'right'
              ? pt.anchorFig.crownX - 4
              : pt.tx + (pt.spk === 'left' ? -12 : 10);
          inner.push(
            wobbledCurvePath(leadStartX, pt.tBot, pt.anchorFig.crownX, pt.anchorFig.crownY, {
              width: 1.95,
              curveX: pt.spk === 'left' ? -5 : 5,
              wobble: 0.45,
              seed: pt.tIdx + 1.6,
            })
          );
        }
      });
    } else {
      const turn = turns[pIdx];
      if (turn) {
        const hasReply = Boolean(turn.reply && figR);
        const fSize = hasReply ? (panelCount === 4 ? 12.6 : 13.6) : (panelCount === 4 ? 13.5 : 14.5);
        const lineStep = hasReply ? (panelCount === 4 ? 14.5 : 15.5) : (panelCount === 4 ? 15.5 : 17);
        const spkIsRight = turn.speaker === 'right' && figR;
        const primaryFig = spkIsRight ? figR : figL;

        const maxPrimaryW = hasReply ? Math.round(panelW * 0.80) : panelW - 20;
        const pLines = wrapTextPx(turn.text, maxPrimaryW, fSize).slice(0, hasReply ? 3 : 4);
        const pxText = hasReply
          ? Math.round(panelW * (spkIsRight ? 0.55 : 0.45))
          : Math.round(panelW / 2);

        pLines.forEach((ln, li) => {
          inner.push(
            `<text x="${pxText}" y="${22 + li * lineStep}" text-anchor="middle" font-size="${fSize}" fill="#0a0a0a">${escapeXml(ln)}</text>`
          );
        });
        const pBot = 14 + pLines.length * lineStep;

        if (hasReply) {
          const replyFig = spkIsRight ? figL : figR;
          // Keep primary speaker's leader strictly in their horizontal lane (left or right)
          const pLeadX = spkIsRight
            ? Math.min(panelW - 20, Math.max(Math.round(panelW * 0.74), primaryFig.crownX))
            : Math.max(20, Math.min(Math.round(panelW * 0.26), primaryFig.crownX));
          if (pBot < primaryFig.crownY - 4) {
            inner.push(
              wobbledCurvePath(pLeadX, pBot, primaryFig.crownX, primaryFig.crownY, {
                width: 1.95,
                curveX: spkIsRight ? 4 : -4,
                wobble: 0.45,
                seed: pIdx + 2.2,
              })
            );
          }

          // Place reply utterance in the replying speaker's opposite horizontal lane so leaders never cross text
          const rY0 = Math.max(76, pBot + 15);
          const maxReplyW = Math.round(panelW * 0.58);
          const rLines = wrapTextPx(turn.reply, maxReplyW, fSize).slice(0, 4);
          const rxText = spkIsRight
            ? Math.round(panelW * 0.34)
            : Math.round(panelW * 0.66);
          rLines.forEach((rln, ri) => {
            inner.push(
              `<text x="${rxText}" y="${rY0 + ri * lineStep}" text-anchor="middle" font-size="${fSize}" fill="#0a0a0a">${escapeXml(rln)}</text>`
            );
          });
          const rBot = rY0 + (rLines.length - 1) * lineStep + 5;
          if (rBot < replyFig.crownY - 4) {
            const rLeadX = spkIsRight
              ? Math.max(22, rxText - 6)
              : Math.min(panelW - 22, rxText + 6);
            inner.push(
              wobbledCurvePath(rLeadX, rBot, replyFig.crownX, replyFig.crownY, {
                width: 1.95,
                curveX: spkIsRight ? -4 : 4,
                wobble: 0.45,
                seed: pIdx + 4.1,
              })
            );
          }
        } else if (pBot < primaryFig.crownY - 4) {
          const startX = Math.round(panelW / 2) + (spkIsRight ? 10 : -10);
          inner.push(
            wobbledCurvePath(startX, pBot, primaryFig.crownX, primaryFig.crownY, {
              width: 1.95,
              curveX: spkIsRight ? 5 : -5,
              wobble: 0.45,
              seed: pIdx + 1.8,
            })
          );
        }
      }
    }

    panelSvgs.push(`<g transform="translate(${px0}, ${py0})">\n    ${inner.join('\n    ')}\n  </g>`);
  }

  return {
    width: totalW,
    height: totalH,
    body: panelSvgs.join('\n  '),
  };
}

function renderAnnotatedChart(spec, helpers) {
  const { escapeXml, wobbledCurvePath, wobbledPolylinePath, wobbledRectPath, arrowheadPaths } = helpers;
  const W = 540;
  const H = 344;
  const ox = 76;
  const oy = 274;
  const topY = 38;
  const rightX = 492;
  const meta = spec.chart_meta || {};
  const variant = meta.curve_style || 'cliff';
  const els = [];

  els.push(wobbledRectPath(4, 4, W - 5, H - 5, { width: 2.2, wobble: 0.5, fill: '#ffffff' }));
  els.push(wobbledCurvePath(ox, oy + 8, ox, topY, { width: 2.1, wobble: 0.6, seed: 1.2 }));
  els.push(arrowheadPaths(ox, oy, ox, topY, 7.0));
  els.push(wobbledCurvePath(ox - 8, oy, rightX, oy, { width: 2.1, wobble: 0.6, seed: 2.4 }));
  els.push(arrowheadPaths(ox, oy, rightX, oy, 7.0));

  [156, 236, 316, 396].forEach(tx => {
    els.push(wobbledCurvePath(tx, oy - 4, tx, oy + 5, { width: 1.8, wobble: 0.2, seed: tx }));
  });
  [218, 160, 102].forEach(ty => {
    els.push(wobbledCurvePath(ox - 5, ty, ox + 4, ty, { width: 1.8, wobble: 0.2, seed: ty }));
  });

  if (variant === 'bimodal') {
    // Bimodal / roulette distribution (e.g. USB-C cables)
    const curvePts = [
      [88, 258],
      [118, 118],
      [148, 254],
      [212, 262],
      [296, 262],
      [372, 258],
      [418, 92],
      [452, 256],
    ];
    els.push(wobbledPolylinePath(curvePts, { width: 2.4, wobble: 0.65 }));
    els.push(wobbledCurvePath(162, 92, 124, 115, { width: 1.8, curveY: -6, wobble: 0.35, seed: 3.3 }));
    els.push(arrowheadPaths(138, 106, 124, 115, 6.0));
    els.push(`<text x="178" y="84" text-anchor="middle" font-size="12.5" fill="#0a0a0a">${escapeXml(meta.left_peak || '5V FAN CABLE FROM 2018')}</text>`);

    els.push(wobbledCurvePath(384, 68, 414, 88, { width: 1.8, curveY: -6, wobble: 0.35, seed: 5.1 }));
    els.push(arrowheadPaths(398, 79, 414, 88, 6.0));
    String(meta.callout || '40GBPS THUNDERBOLT\n(LOOKS IDENTICAL)')
      .split('\n')
      .forEach((ln, idx) => {
        els.push(`<text x="368" y="${42 + idx * 15.5}" text-anchor="middle" font-size="12.5" fill="#0a0a0a">${escapeXml(ln)}</text>`);
      });
  } else if (variant === 'quant_cliff') {
    // High plateau then dramatic quantization cliff
    const curvePts = [
      [88, 94],
      [168, 96],
      [248, 102],
      [318, 118],
      [352, 232],
      [412, 256],
      [468, 260],
    ];
    els.push(wobbledPolylinePath(curvePts, { width: 2.4, wobble: 0.7 }));
    els.push(wobbledCurvePath(318, 118, 464, 124, { width: 1.6, wobble: 0.4, seed: 4.4, dashed: true }));
    els.push(`<text x="412" y="112" text-anchor="middle" font-size="12.5" fill="#2d2d2d">${escapeXml(meta.projected || 'CLAIMED BENCHMARK')}</text>`);
    els.push(wobbledCurvePath(406, 186, 356, 218, { width: 1.85, curveY: -8, wobble: 0.35, seed: 6.2 }));
    els.push(arrowheadPaths(372, 208, 356, 218, 6.0));
      String(meta.callout || 'OUTPUTS ONLY\nMEDIEVAL SOUP RECIPES')
        .split('\n')
        .forEach((ln, idx) => {
          els.push(`<text x="424" y="${168 + idx * 16}" text-anchor="middle" font-size="13" fill="#0a0a0a">${escapeXml(ln)}</text>`);
        });
    } else if (variant === 'hockey_stick') {
      // Flat near-zero baseline followed by sudden vertical spike at the right
      const curvePts = [
        [88, 258],
        [176, 257],
        [264, 256],
        [346, 254],
        [394, 242],
        [424, 148],
        [446, 56],
      ];
      els.push(wobbledPolylinePath(curvePts, { width: 2.4, wobble: 0.68 }));
      els.push(wobbledCurvePath(324, 96, 430, 74, { width: 1.85, curveY: -8, wobble: 0.35, seed: 4.9 }));
      els.push(arrowheadPaths(408, 78, 430, 74, 6.0));
      String(meta.callout || 'SUDDEN PANIC AT 11:58 PM')
        .trim()
        .split('\n')
        .forEach((ln, idx) => {
          els.push(`<text x="212" y="${94 + idx * 16}" text-anchor="middle" font-size="12.5" fill="#0a0a0a">${escapeXml(ln)}</text>`);
        });
      if (meta.left_peak) {
        els.push(`<text x="214" y="240" text-anchor="middle" font-size="12" fill="#2d2d2d">${escapeXml(meta.left_peak)}</text>`);
      }
    } else if (variant === 'inverse_decay') {
      // Steep drop from confident start to asymptotic floor with small Dunning-Kruger bump
      const curvePts = [
        [88, 66],
        [124, 186],
        [168, 236],
        [236, 248],
        [318, 252],
        [404, 246],
        [466, 242],
      ];
      els.push(wobbledPolylinePath(curvePts, { width: 2.4, wobble: 0.68 }));
      els.push(wobbledCurvePath(194, 82, 102, 72, { width: 1.85, curveY: -7, wobble: 0.35, seed: 3.7 }));
      els.push(arrowheadPaths(124, 75, 102, 72, 6.0));
      els.push(`<text x="202" y="86" text-anchor="start" font-size="12.5" fill="#0a0a0a">${escapeXml(meta.projected || 'DAY 1: HOW HARD CAN IT BE?')}</text>`);
      els.push(wobbledCurvePath(346, 192, 308, 244, { width: 1.85, curveY: -6, wobble: 0.35, seed: 5.8 }));
      els.push(arrowheadPaths(320, 228, 308, 244, 6.0));
      String(meta.callout || 'QUIET ACCEPTANCE')
        .split('\n')
        .forEach((ln, idx) => {
          els.push(`<text x="372" y="${168 + idx * 16}" text-anchor="middle" font-size="13" fill="#0a0a0a">${escapeXml(ln)}</text>`);
        });
    } else {
    const curvePts = [
      [86, 250],
      [136, 212],
      [188, 146],
      [234, 108],
      [270, 114],
      [304, 178],
      [342, 238],
      [402, 254],
      [468, 258],
    ];
    els.push(wobbledPolylinePath(curvePts, { width: 2.4, wobble: 0.75 }));
    els.push(wobbledCurvePath(234, 108, 462, 62, { width: 1.7, wobble: 0.45, seed: 5.5, dashed: true }));
    els.push(wobbledCurvePath(346, 134, 280, 116, { width: 1.9, curveY: -10, wobble: 0.4, seed: 7.1 }));
    els.push(arrowheadPaths(302, 118, 280, 116, 6.0));
    els.push(`<text x="382" y="48" text-anchor="middle" font-size="13" fill="#2d2d2d">${escapeXml(meta.projected || 'OPTIMISTIC ROADMAP')}</text>`);
    String(meta.callout || 'POINT WHERE YOU REWRITE\nIT ALL IN A SHELL SCRIPT')
      .split('\n')
      .forEach((ln, idx) => {
        els.push(`<text x="398" y="${152 + idx * 16}" text-anchor="middle" font-size="13" fill="#0a0a0a">${escapeXml(ln)}</text>`);
      });
  }

  els.push(`<text x="${(ox + rightX) / 2}" y="312" text-anchor="middle" font-size="15" fill="#0a0a0a">${escapeXml(meta.x_label || 'EFFORT INVESTED')}</text>`);
  const yMid = (topY + oy) / 2;
  els.push(
    `<text x="36" y="${yMid}" text-anchor="middle" font-size="15" fill="#0a0a0a" transform="rotate(-90 36 ${yMid})">${escapeXml(meta.y_label || 'ACTUAL OUTPUT')}</text>`
  );
  return { width: W, height: H, body: els.join('\n  ') };
}

function renderFlowchartDiagram(spec, helpers) {
  const { escapeXml, wrapTextPx, wobbledCurvePath, wobbledPolylinePath, wobbledRectPath, arrowheadPaths } = helpers;
  const W = 548;
  const H = 356;
  const meta = spec.chart_meta || {};
  const els = [];

  els.push(wobbledRectPath(4, 4, W - 5, H - 5, { width: 2.2, wobble: 0.5, fill: '#ffffff' }));
  els.push(wobbledRectPath(146, 24, 394, 68, { width: 2.0, wobble: 0.5, fill: '#ffffff' }));
  els.push(wobbledCurvePath(270, 68, 270, 104, { width: 2.0, wobble: 0.35, seed: 1.1 }));
  els.push(arrowheadPaths(270, 88, 270, 104, 6.0));

  const diamond = [
    [270, 106],
    [376, 154],
    [270, 202],
    [164, 154],
  ];
  els.push(wobbledPolylinePath(diamond, { width: 2.0, wobble: 0.5, closed: true }));

  els.push(wobbledCurvePath(376, 154, 410, 154, { width: 2.0, wobble: 0.3, seed: 2.2 }));
  els.push(arrowheadPaths(394, 154, 410, 154, 6.0));
  els.push(wobbledRectPath(412, 124, 528, 184, { width: 2.0, wobble: 0.5, fill: '#ffffff' }));

  els.push(wobbledCurvePath(270, 202, 270, 242, { width: 2.0, wobble: 0.35, seed: 3.3 }));
  els.push(arrowheadPaths(270, 226, 270, 242, 6.0));
  els.push(wobbledRectPath(142, 244, 398, 292, { width: 2.0, wobble: 0.5, fill: '#ffffff' }));

  els.push(
    wobbledPolylinePath(
      [
        [142, 268],
        [72, 268],
        [72, 154],
        [161, 154],
      ],
      { width: 1.9, wobble: 0.5 }
    )
  );
  els.push(arrowheadPaths(142, 154, 161, 154, 6.0));

  els.push(`<text x="270" y="51" text-anchor="middle" font-size="14.5" fill="#0a0a0a">${escapeXml(meta.start || 'EVALUATING ARCHITECTURE')}</text>`);
  els.push(`<text x="270" y="159" text-anchor="middle" font-size="13" fill="#0a0a0a">${escapeXml(meta.diamond || 'FITS IN 10 MB?')}</text>`);
  els.push(`<text x="392" y="144" text-anchor="middle" font-size="12.5" fill="#0a0a0a">YES</text>`);
  const yesLines = wrapTextPx(meta.yes_box || 'SHIP ZERO-SERVER WASM', 104, 12.5).slice(0, 3);
  yesLines.forEach((ln, idx) => {
    const yPos = 158 - ((yesLines.length - 1) * 8) + idx * 15;
    els.push(`<text x="470" y="${yPos}" text-anchor="middle" font-size="12.5" fill="#0a0a0a">${escapeXml(ln)}</text>`);
  });
  els.push(`<text x="286" y="225" text-anchor="middle" font-size="12.5" fill="#0a0a0a">NO</text>`);
  els.push(`<text x="270" y="273" text-anchor="middle" font-size="14" fill="#0a0a0a">${escapeXml(meta.no_box || 'QUANTIZE WEIGHTS TO INT8')}</text>`);
  els.push(`<text x="86" y="320" text-anchor="middle" font-size="12.5" fill="#373737">${escapeXml(meta.loop_label || 'REPEAT UNTIL IT FITS')}</text>`);
  return { width: W, height: H, body: els.join('\n  ') };
}
