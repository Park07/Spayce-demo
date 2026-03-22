---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    background: #131315;
    color: #e5e1e4;
    font-family: Inter, sans-serif;
    padding: 56px;
  }
  h1, h2, h3 {
    font-family: "Space Grotesk", sans-serif;
    color: #e5e1e4;
    letter-spacing: -0.03em;
  }
  h1 {
    font-size: 40px;
    margin-bottom: 18px;
  }
  h2 {
    font-size: 30px;
    margin-bottom: 14px;
  }
  p, li {
    font-size: 20px;
    line-height: 1.45;
  }
  strong {
    color: #8ed5ff;
  }
  code {
    color: #ffb95f;
    background: transparent;
  }
  ul {
    margin-top: 10px;
  }
  .small {
    font-size: 14px;
    color: #bdc8d1;
  }
  .accent {
    color: #8ed5ff;
  }
  .warn {
    color: #ffb95f;
  }
---

# SPACEY_OS
## Crowd intelligence before crisis strikes

- Privacy-aware crowd safety intelligence for venues, transit hubs, and mass events
- Existing CCTV in, early warning and operator action out
- Built for live operations, not retrospective analytics

<div class="small">FoundersHack preliminary judging deck draft</div>

---

# The Problem

Crowd incidents are rarely sudden. They build through:

- rising local density
- stalled or conflicting flow
- bottleneck pressure near exits and corridors
- delayed human recognition across too many camera feeds

**Operators have surveillance, but not intelligence.**

---

# Why This Matters

- When operators identify dangerous crowd behavior too late, intervention happens after the safe response window has already narrowed
- That creates injury risk, panic risk, shutdown risk, reputational risk, and legal exposure
- Existing workflows rely on manual monitoring of multiple feeds, radio escalation, and judgment under pressure

**The cost of late detection is nonlinear.**

---

# Why Now

- Venues already have widespread CCTV coverage
- Computer vision is now good enough for live crowd counting, flow analysis, and zone monitoring
- Explainable AI can translate telemetry into operator-readable actions
- In-person events are back, staffing pressure remains high, and safety expectations are rising

<div class="small">Research basis: CDC mass gathering guidance, DHS crowd-analysis technology reports, Intel/WaitTime venue case study, smart stadium market growth data.</div>

---

# Target Customer

Primary users:

- control room operators
- venue operations managers
- security supervisors
- event safety teams

Economic buyers:

- stadium and arena operators
- festival and event organizers
- public transport authorities
- security contractors

---

# Customer Pain

- Too many feeds, not enough operator attention
- Counts and congestion are hard to estimate reliably in real time
- Existing systems show footage, not predicted risk
- Teams need to know **where risk is forming** and **what to do next**

**Customers do not want more video. They want earlier decisions.**

---

# Our Solution

SPACEY turns passive surveillance into an operational early-warning system.

- detects people and tracks movement from live video
- monitors density, bottlenecks, and crowd stress
- highlights risk spatially with bounded overlays
- gives explainable operator-facing alerts and recommended actions

**From camera feed to intervention workflow.**

---

# Product Demo Flow

1. Ingest CCTV or live stream
2. Detect and track people in motion
3. Compute density, flow, and bottleneck metrics
4. Score crowd stress risk in real time
5. Surface operator alerts, zone breakdowns, and recommendations

---

# What We Built

- Running Streamlit application
- Live and uploaded video analysis modes
- Zone editor for exits, entries, and bottlenecks
- Stabilized active-track counting
- Green / yellow / red bounded risk boxes
- Live session overview, alerts, and trend charts
- Explainable AI assistant layer

**This is a working interactive prototype, not just a design mockup.**

---

# Why We’re Different

Most alternatives solve only one slice:

- occupancy counting
- general security analytics
- post-event reporting
- threat screening

SPACEY combines:

- live counting
- zone-based crowd risk
- privacy-aware CV
- explainable intervention support

---

# Defensibility

- Built on top of existing camera infrastructure
- Operational workflow focus creates switching cost
- Historical event data improves thresholds and planning over time
- Explainable outputs help build user trust faster than black-box scoring alone

**We are not replacing cameras or guards. We make existing systems smarter.**

---

# Market

- Smart stadium and venue security spend is already growing strongly
- Crowd-intelligence and occupancy-monitoring solutions are already being adopted in major venues
- Safety, throughput, and staffing efficiency all create budget justification

Near-term beachhead:

- arenas
- stadiums
- festivals
- transit interchanges

---

# Business Model

- annual B2B software subscription per site
- optional per-camera / per-zone tiering
- onboarding and integration fee
- enterprise analytics and support add-ons
- optional on-prem or edge deployment premium

**Recurring operational infrastructure, not a one-off project sale.**

---

# Go-To-Market

- Start with pilot deployments in high-density venues
- Sell into operators already buying CCTV, control-room, or safety systems
- Partner with security integrators and venue-tech channels
- Prove ROI through earlier detection, better staffing deployment, and safer throughput

---

# Why This Can Become a Venture

- Large, recurring B2B safety and operations budget
- Clear wedge into existing infrastructure
- Expands from one site to multi-site deployments
- Strong retention once embedded into operations
- Valuable data layer grows with every event processed

**This can become the intelligence layer between surveillance and intervention.**

---

# Validation Strategy

- Compare system counts against manual review on sample footage
- Test whether alerts arrive early enough to change operator action
- Run structured feedback sessions with venue and security teams
- Tune thresholds by venue type and traffic pattern

**We are testing for operational trust, not just model accuracy.**

---

# Risks And Our Answer

- False positives
  - human-in-the-loop alerts and threshold tuning
- Privacy concerns
  - face blurring and existing-camera-first deployment
- Slow enterprise sales cycles
  - pilot-first approach with integrator support
- Competitive market
  - focus on intervention workflow, not just analytics

---

# Ask

Over the next phase we want to:

- validate with venue and event operators
- refine live risk thresholds using real footage
- tighten deployment and onboarding for pilot customers
- move from demo prototype to field-ready pilot

**We are building the system operators wish they had 30 seconds before a crowd incident escalates.**

---

# Closing

## Crowd disasters are rarely unpredictable.
## They are usually undetected early enough.

**SPACEY gives operators earlier visibility, clearer decisions, and faster intervention.**

<div class="small">Appendix sources available on request: CDC, DHS, Intel/WaitTime, Grand View Research, event industry reports, relevant crowd-analysis research.</div>
