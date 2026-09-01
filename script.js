/* ==========================================
   SPIDEY-SENSE FRONTEND ENGINE
========================================== */


/* Smooth scroll */

function scrollToAnalyzer() {

    document
        .getElementById("analyzer")
        .scrollIntoView({
            behavior: "smooth"
        });

}


/* ==========================================
   TRANSACTION ANALYSIS
========================================== */

function analyzeTransaction() {

    const amount =
        Number(document.getElementById("amount").value);

    const location =
        document.getElementById("location").value;

    const device =
        document.getElementById("device").value;

    const velocity =
        Number(document.getElementById("velocity").value);


    /*
        TEMPORARY DEMO LOGIC

        Later this section will be replaced with:

        fetch("http://localhost:8000/predict")

        and your friend's actual ML model.
    */


    let score = 8;

    let signals = [];


    /* Amount */

    if (amount > 50000) {

        score += 30;

        signals.push(
            "Unusually large transaction amount"
        );

    }

    else if (amount > 10000) {

        score += 15;

        signals.push(
            "Amount above normal baseline"
        );

    }


    /* Device */

    if (device === "New device") {

        score += 25;

        signals.push(
            "New device detected"
        );

    }


    /* Velocity */

    if (velocity >= 5) {

        score += 20;

        signals.push(
            "High transaction velocity"
        );

    }


    /* Location */

    if (
        location.toLowerCase() !== "chennai" &&
        location.toLowerCase() !== "madurai" &&
        location.toLowerCase() !== "coimbatore"
    ) {

        score += 15;

        signals.push(
            "Unusual location"
        );

    }


    score = Math.min(score, 99);


    let probability =
        Math.round(score * 0.97);


    let level;


    if (score <= 30) {

        level = "LOW RISK";

    }

    else if (score <= 60) {

        level = "MEDIUM RISK";

    }

    else if (score <= 80) {

        level = "HIGH RISK";

    }

    else {

        level = "CRITICAL RISK";

    }


    /* ==========================================
       UPDATE UI
    ========================================== */

    document.getElementById("riskDisplay")
        .textContent = score + "%";


    document.getElementById("probability")
        .textContent = probability + "%";


    document.getElementById("score")
        .textContent = score + " / 100";


    document.getElementById("riskLevel")
        .textContent = level;


    /* Signals */

    const signalList =
        document.getElementById("agentSignals");

    signalList.innerHTML = "";


    if (signals.length === 0) {

        signals.push(
            "Transaction matches normal behavior"
        );

    }


    signals.forEach(signal => {

        const li =
            document.createElement("li");

        li.textContent = signal;

        signalList.appendChild(li);

    });


    /* Agent explanation */

    let explanation;


    if (score <= 30) {

        explanation =
            "The transaction closely matches the customer's normal behavioral pattern.";

    }

    else if (score <= 60) {

        explanation =
            "The transaction shows some unusual behavior and should be monitored.";

    }

    else if (score <= 80) {

        explanation =
            "The transaction contains multiple behavioral anomalies requiring additional verification.";

    }

    else {

        explanation =
            "The transaction strongly deviates from the customer's normal behavior across multiple risk signals.";

    }


    document.getElementById("agentText")
        .textContent = explanation;


    document
        .querySelector(".agent-report")
        .scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

}


/* ==========================================
   AI AGENT RECOMMENDATION
========================================== */

function recommendAction() {

    const score =
        Number(
            document
                .getElementById("score")
                .textContent
                .split(" ")[0]
        );


    let recommendation;


    if (score <= 30) {

        recommendation =
            "✓ APPROVE — Transaction appears safe.";

    }

    else if (score <= 60) {

        recommendation =
            "⚠ MONITOR — Flag transaction for review.";

    }

    else if (score <= 80) {

        recommendation =
            "🔐 ADDITIONAL VERIFICATION — Request OTP/device confirmation.";

    }

    else {

        recommendation =
            "🚨 HOLD AND REVIEW — Escalate to fraud analyst.";

    }


    document.getElementById("recommendation")
        .textContent = recommendation;

}


/* ==========================================
   DEMO
========================================== */

function showDemo() {

    document.getElementById("amount").value = 87500;

    document.getElementById("location").value = "Mumbai";

    document.getElementById("device").value = "New device";

    document.getElementById("merchant").value =
        "Unfamiliar merchant";

    document.getElementById("velocity").value = 6;

    document.getElementById("time").value = "10:44";


    scrollToAnalyzer();


    setTimeout(() => {

        analyzeTransaction();

    }, 700);

}