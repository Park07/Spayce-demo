document.addEventListener("DOMContentLoaded", () => {
    const heroRisk = document.getElementById("hero-risk");
    const stressScore = document.getElementById("stress-score");
    const kineticScore = document.getElementById("kinetic-score");
    const vocalScore = document.getElementById("vocal-score");
    const densityScore = document.getElementById("density-score");
    const kineticBar = document.getElementById("kinetic-bar");
    const vocalBar = document.getElementById("vocal-bar");
    const densityBar = document.getElementById("density-bar");
    const frameDensity = document.getElementById("frame-density");
    const frameStatus = document.getElementById("frame-status");
    const frameCallout = document.getElementById("frame-callout");
    const toolbarAlert = document.getElementById("toolbar-alert");
    const terminalLines = document.getElementById("terminal-lines");
    const auditList = document.getElementById("audit-list");

    const dashboardStates = [
        {
            risk: 0.73,
            level: "Elevated",
            stress: 8.4,
            density: 0.73,
            kinetic: 0.82,
            vocal: 0.91,
            overload: 0.65,
            callout: "Stopped flow detected / Exit B",
            toolbar: "Alert: Stopped_Flow_Zone_B",
            frameStatus: "Risk Critical"
        },
        {
            risk: 0.68,
            level: "High watch",
            stress: 7.6,
            density: 0.68,
            kinetic: 0.74,
            vocal: 0.63,
            overload: 0.58,
            callout: "Counterflow detected / Concourse A",
            toolbar: "Alert: Counterflow_Concourse_A",
            frameStatus: "Risk High"
        },
        {
            risk: 0.81,
            level: "Critical",
            stress: 9.1,
            density: 0.81,
            kinetic: 0.88,
            vocal: 0.95,
            overload: 0.76,
            callout: "Pressure spike detected / Gate 3",
            toolbar: "Alert: Pressure_Spike_Gate_3",
            frameStatus: "Risk Critical"
        }
    ];

    const auditTemplates = [
        {
            tag: "AUTO DETECT",
            tone: "audit-tag-blue",
            message: "Anomaly identified: group behavior sync in Quadrant 4."
        },
        {
            tag: "RISK UPGRADE",
            tone: "audit-tag-amber",
            message: "Stress index elevated due to biometric sensor relay."
        },
        {
            tag: "SYSTEM IDLE",
            tone: "",
            message: "Data stream synchronization complete. All nodes active."
        },
        {
            tag: "ROUTE SHIFT",
            tone: "audit-tag-blue",
            message: "Crowd flow nudged toward concourse A to reduce platform pressure."
        },
        {
            tag: "INTERVENTION",
            tone: "audit-tag-amber",
            message: "Operator recommendation queued for managed release protocol."
        }
    ];

    const terminalTemplates = [
        "SYSTEM_SCAN: Completed Sector 7.",
        "RISK_DETECTED: HIGH (0.81) in Zone C.",
        "#ANALYSIS: Density spike at stairwell bottleneck. Pattern matches pre-incident cluster 004-A.",
        "#ACTION: Deploy staff to Exit 3. Trigger redirection audio 02.",
        "AUDIO_MONITOR: Elevated decibel band detected near gate 12.",
        "FLOW_MODEL: Counter-direction vector anomaly building in concourse A."
    ];

    let dashboardIndex = 0;
    let auditIndex = 0;
    let terminalIndex = 0;

    const formatTime = (date) =>
        date.toLocaleTimeString("en-AU", {
            hour12: false,
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        });

    const applyState = (state) => {
        heroRisk.textContent = `RISK SCORE: ${state.risk.toFixed(2)} / ${state.level.toUpperCase()}`;
        stressScore.textContent = state.stress.toFixed(1);
        kineticScore.textContent = state.kinetic.toFixed(2);
        vocalScore.textContent = state.vocal.toFixed(2);
        densityScore.textContent = state.overload.toFixed(2);
        kineticBar.style.width = `${state.kinetic * 100}%`;
        vocalBar.style.width = `${state.vocal * 100}%`;
        densityBar.style.width = `${state.overload * 100}%`;
        frameDensity.textContent = state.density.toFixed(2);
        frameStatus.textContent = state.frameStatus;
        frameCallout.textContent = state.callout;
        toolbarAlert.textContent = state.toolbar;
    };

    const prependAudit = () => {
        const template = auditTemplates[auditIndex % auditTemplates.length];
        auditIndex += 1;

        const entry = document.createElement("article");
        entry.className = "audit-entry is-new";
        entry.innerHTML = `
            <div class="audit-meta">
                <time>${formatTime(new Date())}</time>
                <span class="audit-tag ${template.tone}">${template.tag}</span>
            </div>
            <p>${template.message}</p>
        `;

        auditList.prepend(entry);

        while (auditList.children.length > 4) {
            auditList.removeChild(auditList.lastElementChild);
        }

        window.setTimeout(() => {
            entry.classList.remove("is-new");
        }, 1400);
    };

    const appendTerminalLine = () => {
        const text = terminalTemplates[terminalIndex % terminalTemplates.length];
        terminalIndex += 1;

        const line = document.createElement("p");
        const time = formatTime(new Date());

        if (text.startsWith("#ACTION")) {
            line.innerHTML = `<span class="callout-token">#ACTION</span> ${text.replace("#ACTION: ", "")}`;
        } else if (text.startsWith("#ANALYSIS")) {
            line.className = "warning-text";
            line.textContent = text;
        } else {
            line.innerHTML = `<span class="accent">[${time}]</span> ${text}`;
        }

        terminalLines.appendChild(line);

        while (terminalLines.children.length > 5) {
            terminalLines.removeChild(terminalLines.firstElementChild);
        }
    };

    document.querySelectorAll("[data-scroll-target]").forEach((button) => {
        button.addEventListener("click", () => {
            const target = document.querySelector(button.dataset.scrollTarget);

            if (target) {
                target.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        });
    });

    applyState(dashboardStates[dashboardIndex]);

    window.setInterval(() => {
        dashboardIndex = (dashboardIndex + 1) % dashboardStates.length;
        applyState(dashboardStates[dashboardIndex]);
    }, 4200);

    window.setInterval(prependAudit, 5200);
    window.setInterval(appendTerminalLine, 4200);
});
