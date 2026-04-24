const COIN_COLORS = ["gold", "red", "orange", "yellow", "green", "blue"];
const COIN_TEXT = { gold: "G", red: "R", orange: "O", yellow: "Y", green: "G", blue: "B" };

// DOM Elements
const loginScreen = document.getElementById("login-screen");
const gameUI = document.getElementById("game-ui");
const loginStatus = document.getElementById("login-status");

let socket = null;
let myIndex = null;
let roomName = "";
let playerName = "";

// Game State
let currentState = null;
let selectedStones = { gold: 0, red: 0, orange: 0, yellow: 0, green: 0, blue: 0 }; // Count per color
let lastPhaseLogged = null;
let lastTurnLogged = -1;

// Initialize Connection
document.getElementById("btn-join").addEventListener("click", () => {
    const url = document.getElementById("server-url").value.trim();
    roomName = document.getElementById("room-id").value.trim();
    playerName = document.getElementById("player-name").value.trim();

    if (!url || !roomName || !playerName) {
        loginStatus.textContent = "Please fill all fields.";
        return;
    }

    loginStatus.textContent = "Connecting...";

    socket = io(url, { reconnection: true });

    socket.on("connect", () => {
        loginStatus.textContent = "Connected! Joining room...";
        socket.emit("join_room", { room_id: roomName, player_name: playerName });
    });

    socket.on("player_assigned", (data) => {
        myIndex = data.index;
        loginScreen.classList.add("hidden");
        document.getElementById("game-layout").classList.remove("hidden");
        document.getElementById("my-name").textContent = playerName + " (P" + myIndex + ")";
        addLog(`Joined room as Player ${myIndex}`, "game");
    });

    socket.on("state_update", (state) => {
        currentState = state;
        updateUI();
    });

    socket.on("bid_rejected", (data) => {
        const msg = data.message || data.reason || "Bid rejected.";
        showError(msg);
    });

    socket.on("disconnect", () => {
        loginStatus.textContent = "Disconnected from server.";
        loginScreen.classList.remove("hidden");
        document.getElementById("game-layout").classList.add("hidden");
        addLog("Disconnected from server.", "game");
    });
});

// Update the UI based on server state
function updateUI() {
    if (!currentState) return;

    const phase = currentState.phase || "WAITING";
    const turn = currentState.turn || 0;
    const round = currentState.round || 0;
    const ps = currentState.players || [];

    // Header
    document.getElementById("phase-badge").textContent = phase;
    document.getElementById("round-turn-info").textContent = `Round: ${round} | Turn: ${turn} | Bag: ${currentState.bag_left || 0}`;

    // Status Message
    let msg = "Waiting for players...";
    if (phase === "BIDDING") {
        msg = (currentState.current_bidder === myIndex) ? "👉 YOUR TURN TO BID" : "Waiting for opponent's bid...";
    } else if (phase === "RESULT" || phase === "ROUND_END") {
        msg = "Press OK/Next to continue.";
    } else if (phase === "GAME_END") {
        const info = currentState.game_end_info || {};
        msg = (info.winner === myIndex) ? "🎉 YOU WON!" : "💀 YOU LOST!";
    }
    document.getElementById("status-message").textContent = msg;

    // Check for new phase to log
    const phaseKey = `${round}-${turn}-${phase}`;
    if (lastPhaseLogged !== phaseKey) {
        lastPhaseLogged = phaseKey;
        logPhaseChanges(phase, round, turn, ps);
    }

    // Caretaker (Crown)
    const oppIndex = 1 - myIndex;
    const isMyCaretaker = currentState.caretaker === myIndex;
    const isOppCaretaker = currentState.caretaker === oppIndex;
    document.getElementById("my-crown").classList.toggle("hidden", !isMyCaretaker);
    document.getElementById("opp-crown").classList.toggle("hidden", !isOppCaretaker);

    // Update Opponent
    if (oppIndex >= 0 && oppIndex < ps.length) {
        const opp = ps[oppIndex];
        document.getElementById("opp-name").textContent = opp.name + " (P" + oppIndex + ")";
        document.getElementById("opp-score").textContent = opp.score || 0;
        document.getElementById("opp-hand-count").textContent = opp.hand_count || 0;

        const badge = document.getElementById("opp-badge");
        if (phase === "BIDDING") {
            if (opp.bid_submitted) {
                badge.textContent = "Bid Submitted";
                badge.className = "status-badge ready";
            } else if (currentState.current_bidder === oppIndex) {
                badge.textContent = "Thinking...";
                badge.className = "status-badge bidding";
            } else {
                badge.textContent = "Waiting Turn";
                badge.className = "status-badge waiting";
            }
        } else if (phase === "RESULT" || phase === "ROUND_END") {
            if (opp.ok_ready) {
                badge.textContent = "Ready";
                badge.className = "status-badge ready";
            } else {
                badge.textContent = "Reading...";
                badge.className = "status-badge waiting";
            }
        } else {
            badge.textContent = "Done";
            badge.className = "status-badge waiting";
        }
    }

    // Update Myself
    if (myIndex >= 0 && myIndex < ps.length) {
        const me = ps[myIndex];
        document.getElementById("my-score").textContent = me.score || 0;

        // Reset selection if phase changed or turn changed (simplified: reset every update)
        selectedStones = { gold: 0, red: 0, orange: 0, yellow: 0, green: 0, blue: 0 };
        updateSelectedPreview();
        renderHand(me.hand || []);
    }

    // Trash Board
    renderTrash(currentState.trash || {});

    // Offer Board
    renderStones("offer-stones", currentState.offer || []);

    // Controls
    const btnSubmit = document.getElementById("btn-submit");
    const btnNext = document.getElementById("btn-next");
    const btnRestart = document.getElementById("btn-restart");

    btnSubmit.classList.add("hidden");
    btnNext.classList.add("hidden");
    btnRestart.classList.add("hidden");

    if (phase === "BIDDING" && currentState.current_bidder === myIndex && !ps[myIndex].bid_submitted) {
        btnSubmit.classList.remove("hidden");
        // Disable submit if selection is empty (unless offer has colors) - simplified logic: let server reject invalid bids
        btnSubmit.classList.remove("disabled");
    } else if (phase === "RESULT" || phase === "ROUND_END") {
        if (!ps[myIndex].ok_ready) {
            btnNext.classList.remove("hidden");
        }
    } else if (phase === "GAME_END") {
        btnRestart.classList.remove("hidden");
    }
}

// Render Trash Bars
function renderTrash(trashMap) {
    const container = document.getElementById("trash-container");
    container.innerHTML = "";

    COIN_COLORS.forEach(c => {
        const count = Math.min(trashMap[c] || 0, 6); // max 6
        const pct = (count / 6) * 100;

        const item = document.createElement("div");
        item.className = "trash-item";
        item.innerHTML = `
            <span class="trash-label t-${c}">${COIN_TEXT[c]}</span>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${pct}%; background: var(--${c});"></div>
            </div>
            <span>${count}/6</span>
        `;
        container.appendChild(item);
    });
}

// Render Stones (non-interactive)
function renderStones(containerId, stonesList) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    stonesList.forEach(s => {
        const c = s.toLowerCase();
        const el = document.createElement("div");
        el.className = `stone ${c}`;
        el.textContent = COIN_TEXT[c] || "?";
        container.appendChild(el);
    });
}

// Render Player Hand (interactive)
function renderHand(stonesList) {
    const container = document.getElementById("my-hand");
    container.innerHTML = "";

    // Count how many we have of each color
    const handCounts = { gold: 0, red: 0, orange: 0, yellow: 0, green: 0, blue: 0 };
    stonesList.forEach(s => {
        const c = s.toLowerCase();
        if (handCounts[c] !== undefined) handCounts[c]++;
    });

    const offerColors = new Set((currentState.offer || []).map(s => s.toLowerCase()));

    COIN_COLORS.forEach(c => {
        if (handCounts[c] > 0) {
            const maxVal = handCounts[c];
            const group = document.createElement("div");
            group.className = "hand-group";

            group.innerHTML = `
                <div class="stone ${c}">${COIN_TEXT[c]}</div>
                <div class="hand-count-text">Total: ${maxVal}</div>
                <div class="bid-controls">
                    <button class="bid-btn btn-minus" disabled>-</button>
                    <span class="bid-amount">0</span>
                    <button class="bid-btn btn-plus">+</button>
                </div>
            `;

            const btnMinus = group.querySelector(".btn-minus");
            const btnPlus = group.querySelector(".btn-plus");
            const amountSpan = group.querySelector(".bid-amount");
            const isOffered = offerColors.has(c);

            const updateButtons = () => {
                const val = selectedStones[c];
                amountSpan.textContent = val;
                btnMinus.disabled = (val <= 0);
                btnPlus.disabled = isOffered || (val >= maxVal);
                updateSelectedPreview();
            };

            btnMinus.addEventListener("click", () => {
                if (selectedStones[c] > 0) {
                    selectedStones[c]--;
                    updateButtons();
                }
            });

            btnPlus.addEventListener("click", () => {
                if (!isOffered && selectedStones[c] < maxVal) {
                    selectedStones[c]++;
                    updateButtons();
                }
            });

            updateButtons();
            container.appendChild(group);
        }
    });
}

function updateSelectedPreview() {
    let total = 0;
    const parts = [];

    COIN_COLORS.forEach(c => {
        const n = selectedStones[c];
        if (n > 0) {
            total += n;
            parts.push(`${c.toUpperCase()}x${n}`);
        }
    });

    if (total === 0) {
        document.getElementById("selected-preview").textContent = "0 stones (Pass)";
    } else {
        document.getElementById("selected-preview").textContent = parts.join(" ");
    }
}

function showError(msg) {
    const err = document.getElementById("action-error");
    err.textContent = msg;
    setTimeout(() => { err.textContent = ""; }, 3000);
}

// Button Events
document.getElementById("btn-submit").addEventListener("click", () => {
    if (!currentState || myIndex === null) return;

    const stonesToBid = [];
    COIN_COLORS.forEach(c => {
        const n = selectedStones[c];
        for (let i = 0; i < n; i++) {
            stonesToBid.push(c);
        }
    });

    socket.emit("submit_bid", { room_id: roomName, stones: stonesToBid });
    document.getElementById("btn-submit").classList.add("disabled");
});

document.getElementById("btn-next").addEventListener("click", () => {
    socket.emit("proceed_phase", { room_id: roomName });
    document.getElementById("btn-next").classList.add("hidden");
});

document.getElementById("btn-restart").addEventListener("click", () => {
    socket.emit("restart_game", { room_id: roomName });
});

// Logs and Results
function addLog(message, type = "normal") {
    const container = document.getElementById("log-container");
    const el = document.createElement("div");
    el.className = `log-entry ${type}`;

    const now = new Date();
    const timeStr = now.getHours().toString().padStart(2, '0') + ":" + now.getMinutes().toString().padStart(2, '0');

    el.innerHTML = `<div class="log-time">${timeStr}</div><div>${message}</div>`;
    // Prepend to show newest at the top
    container.prepend(el);
}

const LOG_TEXT = { gold: "g", red: "R", orange: "O", yellow: "Y", green: "G", blue: "B" };

function summarizeStones(stonesList) {
    if (!stonesList || stonesList.length === 0) return "-";
    const counts = {};
    stonesList.forEach(s => {
        const c = s.toLowerCase();
        counts[c] = (counts[c] || 0) + 1;
    });
    return COIN_COLORS.filter(c => counts[c]).map(c => `<span class="t-${c}"><b>${LOG_TEXT[c]}</b></span><span style="color: white;">x${counts[c]}</span>`).join(" ");
}

function getBidsArray(lastResult) {
    const bids = lastResult.bids_by_player || lastResult.bids_map || lastResult.bids_per_player;
    if (Array.isArray(bids)) {
        return [bids[0] || [], bids[1] || []];
    } else if (bids && typeof bids === 'object') {
        return [bids[0] || bids["0"] || [], bids[1] || bids["1"] || []];
    }
    return [[], []];
}

function logPhaseChanges(phase, round, turn, ps) {
    if (phase === "RESULT") {
        const lr = currentState.last_result || {};
        const wIdx = lr.winner;

        const offerStonesObj = lr.offer_stones || currentState.last_offer || [];
        const offerStonesStr = summarizeStones(offerStonesObj);

        const bids = getBidsArray(lr);
        const myBidObj = bids[myIndex] || [];
        const oppBidObj = bids[1 - myIndex] || [];
        const myBidStr = summarizeStones(myBidObj);
        const oppBidStr = summarizeStones(oppBidObj);

        if (wIdx !== undefined && wIdx !== null) {
            let winnerName = (wIdx === myIndex) ? "You" : (ps[wIdx] ? ps[wIdx].name : `P${wIdx}`);
            let loserName = (wIdx === myIndex) ? (ps[1 - myIndex] ? ps[1 - myIndex].name : `Opponent`) : "You";

            const winnerBidStr = (wIdx === myIndex) ? myBidStr : oppBidStr;
            const loserBidStr = (wIdx === myIndex) ? oppBidStr : myBidStr;

            const msg = `<b>${winnerName} WON</b><br>` +
                `<span style="margin-left: 8px;">${winnerName} bet: ${winnerBidStr} , Gained: ${offerStonesStr}</span><br>` +
                `<span style="margin-left: 8px;">${loserName} bet: ${loserBidStr}</span>`;
            const logType = (wIdx === myIndex) ? "result-win" : "result-loss";
            addLog(`[R${round} T${turn}]<br>${msg}`, logType);
        } else {
            // Draw
            const msg = `<b>DRAW</b><br>` +
                `<span style="margin-left: 8px;">You bet: ${myBidStr}</span><br>` +
                `<span style="margin-left: 8px;">Opponent bet: ${oppBidStr}</span><br>` +
                `<span style="margin-left: 8px;">Offer Trashed: ${offerStonesStr}</span>`;
            addLog(`[R${round} T${turn}]<br>${msg}`, "result-draw");
        }

    } else if (phase === "ROUND_END") {
        const info = currentState.round_end_info || {};
        const adds = info.adds || [];
        const oppIndex = 1 - myIndex;
        const myAdd = adds[myIndex] || 0;
        const oppAdd = adds[oppIndex] || 0;

        let oppName = ps[oppIndex] ? ps[oppIndex].name : `Opponent`;
        const myAddStr = myAdd > 0 ? `+${myAdd}` : `${myAdd}`;
        const oppAddStr = oppAdd > 0 ? `+${oppAdd}` : `${oppAdd}`;

        // Get hands
        const myHand = summarizeStones(ps[myIndex].hand || []);
        const oppHand = summarizeStones(ps[oppIndex].hand || []);

        // Get trash
        const trashObj = currentState.trash || {};
        const trashStones = [];
        Object.keys(trashObj).forEach(c => {
            for (let i = 0; i < trashObj[c]; i++) trashStones.push(c);
        });
        const trashStr = summarizeStones(trashStones);

        // Get ranked colors
        const rankedArr = info.ranked || [];
        const rankedStr = rankedArr.map(x => `<span class="t-${x[0]}"><b>${LOG_TEXT[x[0]]}</b></span>`).join(" > ");

        const msg = `<b>ROUND END</b><br>` +
            `<span style="margin-left: 8px;">You got : ${myAddStr} pt</span>` +
            `<span style="margin-left: 8px;">${oppName} got : ${oppAddStr} pt</span>`;

        addLog(`[R${round} END]<br>${msg}`, "round");
    } else if (phase === "GAME_END") {
        const info = currentState.game_end_info || {};
        const wIdx = info.winner;
        let winnerName = (wIdx === myIndex) ? "You" : (ps[wIdx] ? ps[wIdx].name : `P${wIdx}`);

        addLog(`<b>GAME OVER</b> - ${winnerName} won the game!`, "game");
    }
}
