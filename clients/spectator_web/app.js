const COLORS = ["gold", "red", "orange", "yellow", "green", "blue"];
const LABEL = { gold: "g", red: "R", orange: "O", yellow: "Y", green: "G", blue: "B" };

let socket = null;
let roomId = "";
let history = [];
let lastState = null;
let chartResetSeq = 0;

const $ = (id) => document.getElementById(id);

$("join-btn").addEventListener("click", join);

function join() {
    const url = $("server-url").value.trim();
    roomId = $("room-id").value.trim() || "room1";
    const name = $("spectator-name").value.trim() || "Spectator";

    if (!url) {
        $("join-status").textContent = "Server URL is required";
        return;
    }

    $("join-status").textContent = "Connecting...";
    socket = io(url, { reconnection: true });

    socket.on("connect", () => {
        socket.emit("join_room", { room_id: roomId, player_name: name, spectator: true });
        $("join-status").textContent = "Joining...";
    });

    socket.on("player_assigned", () => {
        $("join-panel").classList.add("hidden");
        $("watch-panel").classList.remove("hidden");
    });

    socket.on("state_update", (state) => renderState(state || {}));
    socket.on("action", (entry) => pushHistory(entry));
    socket.on("disconnect", () => {
        $("status-text").textContent = "Disconnected";
    });
    socket.on("connect_error", (err) => {
        $("join-status").textContent = `Connection failed: ${err.message || err}`;
    });
}

function renderState(state) {
    lastState = state;
    if (Array.isArray(state.action_log) && history.length === 0) {
        history = state.action_log.slice(-80);
    }
    syncChartReset(state);

    $("room-title").textContent = `Room ${state.room_id || roomId || "-"}`;
    $("status-text").textContent = state.status || "-";
    $("phase").textContent = state.phase || "WAITING";
    $("round-turn").textContent = `R${state.round || 0} T${state.turn || 0}`;
    $("bag").textContent = `Bag ${state.bag_left || 0}`;

    renderPlayer("p0", 0, state);
    renderPlayer("p1", 1, state);
    renderStones($("offer"), safeList(state.offer));
    renderResult(state);
    renderScoreChart(state);
    renderTrash(state.trash || {}, Number(state.trash_limit || 6));
    renderConnections(state.connections || []);
    renderHistory();
}

function renderPlayer(id, index, state) {
    const players = state.players || [];
    const player = players[index] || {};
    const isCaretaker = state.caretaker === index;
    const isBidder = state.current_bidder === index && state.phase === "BIDDING";
    const flags = [
        isCaretaker ? "CT" : "",
        isBidder ? "BID" : "",
        player.bid_submitted ? "DONE" : "",
        player.ok_ready ? "OK" : "",
    ].filter(Boolean).join(" ");

    const root = $(id);
    root.innerHTML = "";
    root.append(
        el("div", "player-head", [
            el("div", "name", `${isCaretaker ? "CT " : ""}P${index} ${player.name || "-"}`),
            el("div", "score", `${flags ? `${flags} ` : ""}${player.score || 0}pt`),
        ]),
        labeledStones("Hand", safeList(player.hand)),
    );
}

function renderResult(state) {
    const lr = state.last_result || {};
    const box = $("result");
    box.innerHTML = "";

    if (!Object.keys(lr).length) {
        box.textContent = "-";
        return;
    }

    if (lr.winner === null || lr.winner === undefined) {
        box.append(el("div", "", "No bid / offer to trash"));
        box.append(labeledStones("Offer", safeList(lr.offer || state.last_offer)));
        return;
    }

    const winner = Number(lr.winner);
    const loser = lr.loser === undefined || lr.loser === null ? 1 - winner : Number(lr.loser);
    box.append(
        el("div", "", `Winner: P${winner} ${lr.winner_name || ""}`),
        el("div", "", `Loser: P${loser} ${lr.loser_name || ""}`),
        labeledStones("Win Bid", bestBid(lr, winner)),
        labeledStones("Lose Bid", bestBid(lr, loser)),
        labeledStones("Offer", safeList(lr.offer || state.last_offer)),
    );
}

function renderScoreChart(state) {
    const root = $("score-chart");
    root.innerHTML = "";

    const resetSeq = history.reduce((last, entry, index) => {
        if (!entry || (entry.kind !== "game_end" && entry.kind !== "restart_game")) return last;
        return Math.max(last, actionSeq(entry, index));
    }, chartResetSeq);
    const activeHistory = state.phase === "GAME_END"
        ? []
        : history.filter((entry, index) => actionSeq(entry, index) > resetSeq);
    const scorePoints = activeHistory
        .filter((entry) => entry && entry.kind === "resolve_after")
        .map((entry) => entry.payload || {})
        .filter((payload) => Array.isArray(payload.score) && payload.score.length >= 2)
        .map((payload) => ({
            label: `R${payload.round || "?"} T${payload.turn || "?"}`,
            turn: Math.max(1, Number(payload.turn || 1)),
            p0: Number(payload.score[0] || 0),
            p1: Number(payload.score[1] || 0),
        }))
        .slice(-30);

    if (!scorePoints.length) {
        const players = state.players || [];
        const p0 = players[0] ? Number(players[0].score || 0) : 0;
        const p1 = players[1] ? Number(players[1].score || 0) : 0;
        root.append(makeScoreSvg([{ label: "T1", turn: 1, p0: 0, p1: 0 }], 1, 1));
        root.append(el("div", "score-legend", [
            el("span", "p0-text", `P0 ${p0}pt`),
            el("span", "p1-text", `P1 ${p1}pt`),
            el("span", "", "T1"),
        ]));
        return;
    }

    const maxTurn = Math.max(1, Number(state.turn || 1), ...scorePoints.map((point) => point.turn));
    const maxScore = Math.max(1, ...scorePoints.flatMap((point) => [point.p0, point.p1]));
    const points = [{ label: "T1", turn: 1, p0: 0, p1: 0 }, ...scorePoints];
    const last = scorePoints[scorePoints.length - 1];

    root.append(
        makeScoreSvg(points, maxTurn, maxScore),
        el("div", "score-legend", [
            el("span", "p0-text", `P0 ${last.p0}pt`),
            el("span", "p1-text", `P1 ${last.p1}pt`),
            el("span", "", last.label),
        ]),
    );
}

function syncChartReset(state) {
    const lastResult = state.last_result || {};
    const isNewGameStart = state.phase === "BIDDING"
        && Number(state.round || 0) === 1
        && Number(state.turn || 0) === 1
        && !Object.keys(lastResult).length;

    if (isNewGameStart) {
        chartResetSeq = Math.max(chartResetSeq, maxHistorySeq());
    }
}

function maxHistorySeq() {
    return history.reduce((maxSeq, entry, index) => Math.max(maxSeq, actionSeq(entry, index)), 0);
}

function actionSeq(entry, index) {
    return Number(entry && entry.seq ? entry.seq : index + 1);
}

function makeScoreSvg(points, maxTurn, maxScore) {
    const w = 260;
    const h = 54;
    const padL = 24;
    const padR = 6;
    const padT = 6;
    const padB = 16;
    const xFor = (turn) => maxTurn === 1
        ? padL
        : padL + ((turn - 1) / (maxTurn - 1)) * (w - padL - padR);
    const yFor = (score) => h - padB - (score / maxScore) * (h - padT - padB);
    const line = (key) => points.map((point) => `${xFor(point.turn).toFixed(1)},${yFor(point[key]).toFixed(1)}`).join(" ");

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.innerHTML = `
        <line class="score-axis" x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}"></line>
        <line class="score-axis" x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}"></line>
        <text class="score-axis-label" x="1" y="${h - padB + 3}">0</text>
        <text class="score-axis-label" x="1" y="${padT + 4}">${maxScore}</text>
        <text class="score-axis-label" x="${padL - 3}" y="${h - 3}">T1</text>
        <text class="score-axis-label" x="${w - 26}" y="${h - 3}">T${maxTurn}</text>
        <polyline class="score-line p0" points="${line("p0")}"></polyline>
        <polyline class="score-line p1" points="${line("p1")}"></polyline>
    `;
    return svg;
}

function renderTrash(trash, limit) {
    const root = $("trash");
    root.innerHTML = "";
    COLORS.forEach((color) => {
        const count = Number(trash[color] || 0);
        const row = el("div", "trash-row");
        row.append(
            stoneCount(color, count),
            el("div", "bar", [el("span", color)]),
            el("span", "", `/${limit}`),
        );
        row.querySelector(".bar > span").style.width = `${Math.min(100, (count / limit) * 100)}%`;
        root.append(row);
    });
}

function renderConnections(connections) {
    const root = $("connections");
    root.innerHTML = "";
    if (!connections.length) {
        root.append(el("div", "", "-"));
        return;
    }
    connections.forEach((c) => {
        root.append(el("div", "", `${c.role || "?"}: ${c.name || "Unknown"} (${c.connected_for_sec || 0}s)`));
    });
}

function pushHistory(entry) {
    history.push(entry);
    if (history.length > 160) history = history.slice(-160);
    renderHistory();
    if (lastState) renderScoreChart(lastState);
}

function renderHistory() {
    const root = $("history");
    root.innerHTML = "";
    const lines = history
        .filter((entry) => entry && entry.kind === "resolve_after")
        .slice(-40)
        .reverse()
        .map(historyLine);

    if (!lines.length) {
        root.append(el("div", "empty", "No results yet"));
        return;
    }
    lines.forEach((line) => root.append(el("div", "history-line", line)));
}

function historyLine(entry) {
    const p = entry.payload || {};
    const lr = p.last_result || {};
    if (p.winner === null || p.winner === undefined) {
        return `R${p.round} T${p.turn}: no bid`;
    }
    const winner = Number(p.winner);
    const loser = lr.loser === undefined || lr.loser === null ? 1 - winner : Number(lr.loser);
    return `R${p.round} T${p.turn}: P${winner} ${summary(bestBid(lr, winner))} / P${loser} ${summary(bestBid(lr, loser))}`;
}

function labeledStones(label, stones) {
    const wrap = el("div", "row");
    wrap.append(el("h2", "", label));
    const strip = el("div", "stones");
    renderStones(strip, stones);
    wrap.append(strip);
    return wrap;
}

function renderStones(root, stones) {
    root.innerHTML = "";
    const counts = countStones(stones);
    const visibleColors = COLORS.filter((color) => counts[color]);
    if (!visibleColors.length) {
        root.append(el("span", "empty", "-"));
        return;
    }
    visibleColors.forEach((color) => {
        root.append(stoneCount(color, counts[color]));
    });
}

function stoneCount(color, count) {
    return el("span", "stone-item", [
        el("span", `stone-symbol ${color}`, LABEL[color] || "?"),
        el("span", "stone-count", `x${count}`),
    ]);
}

function bestBid(lastResult, playerIndex) {
    if (Number(lastResult.winner) === playerIndex && Array.isArray(lastResult.winner_bid)) {
        return safeList(lastResult.winner_bid);
    }
    if (Number(lastResult.loser) === playerIndex && Array.isArray(lastResult.loser_bid)) {
        return safeList(lastResult.loser_bid);
    }
    const bids = lastResult.bids_by_player;
    if (Array.isArray(bids)) return safeList(bids[playerIndex]);
    if (bids && typeof bids === "object") return safeList(bids[playerIndex] || bids[String(playerIndex)]);
    return [];
}

function summary(stones) {
    const counts = countStones(stones);
    return COLORS.filter((color) => counts[color])
        .map((color) => `${LABEL[color]}x${counts[color]}`)
        .join(" ") || "-";
}

function countStones(stones) {
    const counts = {};
    stones.forEach((stone) => {
        const color = String(stone).toLowerCase();
        counts[color] = (counts[color] || 0) + 1;
    });
    return counts;
}

function safeList(value) {
    return Array.isArray(value) ? value.map((item) => String(item).toLowerCase()) : [];
}

function el(tag, className = "", content = "") {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (Array.isArray(content)) {
        node.append(...content);
    } else if (content !== "") {
        node.textContent = content;
    }
    return node;
}
