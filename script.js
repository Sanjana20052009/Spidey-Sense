/* =========================================================
   SPIDEY-SENSE FRONTEND ENGINE
   Connected to FastAPI + XGBoost ML backend
   ========================================================= */


/* =========================================================
   API CONFIGURATION
   ========================================================= */

const API_BASE_URL = "";


/* =========================================================
   SMOOTH SCROLL
   ========================================================= */

function scrollToAnalyzer() {

    const analyzer = document.getElementById("analyzer");

    if (analyzer) {
        analyzer.scrollIntoView({
            behavior: "smooth"
        });
    }
}


/* =========================================================
   SYSTEM STATUS
   ========================================================= */

async function checkBackendHealth() {

    try {

        const response = await fetch(
            `${API_BASE_URL}/api/health`
        );

        if (!response.ok) {
            throw new Error("Backend unavailable");
        }

        const data = await response.json();

        console.log(
            "Spidey-Sense backend:",
            data
        );

        updateSystemStatus(
            data.model_loaded
        );

    } catch (error) {

        console.error(
            "Backend health check failed:",
            error
        );

        updateSystemStatus(false);
    }
}


function updateSystemStatus(modelOnline) {

    const status = document.querySelector(
        ".status"
    );

    if (!status) {
        return;
    }

    if (modelOnline) {

        status.innerHTML =
            "<span></span> SYSTEM ONLINE";

        status.style.color = "#38e58c";

    } else {

        status.innerHTML =
            "<span></span> ML OFFLINE";

        status.style.color = "#ff3344";
    }
}


/* =========================================================
   COLLECT FORM DATA
   ========================================================= */

function getTransactionData() {

    const amount =
        Number(
            document.getElementById(
                "amount"
            ).value
        );

    const location =
        document.getElementById(
            "location"
        ).value.trim();

    const device =
        document.getElementById(
            "device"
        ).value;

    const merchant =
        document.getElementById(
            "merchant"
        ).value.trim();

    const velocity =
        Number(
            document.getElementById(
                "velocity"
            ).value
        );

    const time =
        document.getElementById(
            "time"
        ).value;


    return {

        amount: amount,

        location: location,

        device: device,

        merchant: merchant,

        velocity: velocity,

        time: time
    };
}


/* =========================================================
   VALIDATION
   ========================================================= */

function validateTransaction(data) {

    if (
        !Number.isFinite(data.amount) ||
        data.amount < 0
    ) {

        alert(
            "Please enter a valid transaction amount."
        );

        return false;
    }


    if (!data.location) {

        alert(
            "Please enter a transaction location."
        );

        return false;
    }


    if (!data.merchant) {

        alert(
            "Please enter a merchant."
        );

        return false;
    }


    if (
        !Number.isFinite(data.velocity) ||
        data.velocity < 0
    ) {

        alert(
            "Please enter a valid transaction velocity."
        );

        return false;
    }


    if (!data.time) {

        alert(
            "Please enter a transaction time."
        );

        return false;
    }


    return true;
}


/* =========================================================
   LOADING STATE
   ========================================================= */

function setAnalyzingState(isAnalyzing) {

    const button =
        document.querySelector(
            ".analyze-btn"
        );

    if (!button) {
        return;
    }


    if (isAnalyzing) {

        button.disabled = true;

        button.innerHTML =
            "<span>🕷</span> ANALYZING...";

        button.style.opacity = "0.6";

    } else {

        button.disabled = false;

        button.innerHTML =
            "<span>🕷</span> ANALYZE TRANSACTION";

        button.style.opacity = "1";
    }
}


/* =========================================================
   UPDATE LIVE SYSTEM STATUS
   ========================================================= */

function updateAnalysisStatus() {

    const waiting =
        document.querySelector(
            ".waiting"
        );

    if (waiting) {

        waiting.textContent =
            "COMPLETE";

        waiting.style.color =
            "#38e58c";
    }
}


/* =========================================================
   MAIN TRANSACTION ANALYSIS
   ========================================================= */

async function analyzeTransaction() {

    const transaction =
        getTransactionData();


    if (!validateTransaction(transaction)) {
        return;
    }


    setAnalyzingState(true);


    try {

        console.log(
            "Sending transaction to ML backend:",
            transaction
        );


        const response =
            await fetch(
                `${API_BASE_URL}/api/predict`,
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            transaction
                        )
                }
            );


        const result =
            await response.json();


        if (!response.ok) {

            throw new Error(
                result.detail ||
                "Prediction failed."
            );
        }


        console.log(
            "Spidey-Sense prediction:",
            result
        );


        updateDashboard(
            result
        );


        updateAnalysisStatus();


        const report =
            document.querySelector(
                ".agent-report"
            );


        if (report) {

            report.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });
        }


    } catch (error) {

        console.error(
            "Spidey-Sense error:",
            error
        );


        alert(
            "Unable to contact the Spidey-Sense ML backend.\n\n" +
            error.message +
            "\n\nMake sure you started app.py."
        );

    } finally {

        setAnalyzingState(false);
    }
}


/* =========================================================
   UPDATE DASHBOARD
   ========================================================= */

function updateDashboard(result) {

    const prediction =
        result.prediction || {};

    const signals =
        result.signals || [];

    const explanation =
        result.explanation || "";


    const riskScore =
        Number(
            prediction.risk_score || 0
        );


    const probability =
        Number(
            prediction.fraud_probability || 0
        );


    const riskLevel =
        prediction.risk_level ||
        "UNKNOWN";


    /* -----------------------------------------------------
       Risk display
       ----------------------------------------------------- */

    const riskDisplay =
        document.getElementById(
            "riskDisplay"
        );


    if (riskDisplay) {

        riskDisplay.textContent =
            `${riskScore}%`;
    }


    /* -----------------------------------------------------
       Probability
       ----------------------------------------------------- */

    const probabilityElement =
        document.getElementById(
            "probability"
        );


    if (probabilityElement) {

        probabilityElement.textContent =
            `${Math.round(probability)}%`;
    }


    /* -----------------------------------------------------
       Score
       ----------------------------------------------------- */

    const scoreElement =
        document.getElementById(
            "score"
        );


    if (scoreElement) {

        scoreElement.textContent =
            `${riskScore} / 100`;
    }


    /* -----------------------------------------------------
       Risk level
       ----------------------------------------------------- */

    const riskLevelElement =
        document.getElementById(
            "riskLevel"
        );


    if (riskLevelElement) {

        riskLevelElement.textContent =
            riskLevel;
    }


    /* -----------------------------------------------------
       Agent explanation
       ----------------------------------------------------- */

    const agentText =
        document.getElementById(
            "agentText"
        );


    if (agentText) {

        agentText.textContent =
            explanation;
    }


    /* -----------------------------------------------------
       Signals
       ----------------------------------------------------- */

    const signalList =
        document.getElementById(
            "agentSignals"
        );


    if (signalList) {

        signalList.innerHTML = "";


        if (signals.length === 0) {

            const li =
                document.createElement(
                    "li"
                );

            li.textContent =
                "Transaction matches normal behavior";

            signalList.appendChild(li);

        } else {

            signals.forEach(
                function(signal) {

                    const li =
                        document.createElement(
                            "li"
                        );

                    li.textContent =
                        signal;

                    signalList.appendChild(
                        li
                    );
                }
            );
        }
    }


    /* -----------------------------------------------------
       Recommendation
       ----------------------------------------------------- */

    const recommendation =
        document.getElementById(
            "recommendation"
        );


    if (recommendation) {

        recommendation.textContent =
            "";
    }


    /* -----------------------------------------------------
       Update result cards
       ----------------------------------------------------- */

    updateResultCards(
        riskScore
    );
}


/* =========================================================
   RESULT CARD UPDATE
   ========================================================= */

function updateResultCards(score) {

    const normal =
        document.querySelector(
            ".transaction-result.normal"
        );

    const suspicious =
        document.querySelector(
            ".transaction-result.suspicious"
        );


    if (score <= 30) {

        if (normal) {

            normal.style.opacity = "1";
            normal.style.transform =
                "scale(1.02)";
        }


        if (suspicious) {

            suspicious.style.opacity =
                "0.45";

            suspicious.style.transform =
                "scale(1)";
        }

    } else {

        if (normal) {

            normal.style.opacity =
                "0.45";

            normal.style.transform =
                "scale(1)";
        }


        if (suspicious) {

            suspicious.style.opacity =
                "1";

            suspicious.style.transform =
                "scale(1.02)";
        }
    }
}


/* =========================================================
   AI AGENT RECOMMENDATION
   ========================================================= */

async function recommendAction() {

    const scoreElement =
        document.getElementById(
            "score"
        );


    if (!scoreElement) {
        return;
    }


    const score =
        Number(
            scoreElement.textContent
                .split(" ")[0]
        );


    let recommendation;


    if (score <= 30) {

        recommendation =
            "✓ APPROVE — Transaction appears safe.";

    } else if (score <= 60) {

        recommendation =
            "⚠ MONITOR — Flag transaction for review.";

    } else if (score <= 80) {

        recommendation =
            "🔐 ADDITIONAL VERIFICATION — Request OTP/device confirmation.";

    } else {

        recommendation =
            "🚨 HOLD AND REVIEW — Escalate to fraud analyst.";
    }


    const recommendationElement =
        document.getElementById(
            "recommendation"
        );


    if (recommendationElement) {

        recommendationElement.textContent =
            recommendation;
    }
}


/* =========================================================
   DEMO TRANSACTION
   ========================================================= */

function showDemo() {

    document.getElementById(
        "amount"
    ).value = 87500;


    document.getElementById(
        "location"
    ).value = "Mumbai";


    document.getElementById(
        "device"
    ).value = "New device";


    document.getElementById(
        "merchant"
    ).value =
        "Unfamiliar merchant";


    document.getElementById(
        "velocity"
    ).value = 6;


    document.getElementById(
        "time"
    ).value = "10:44";


    scrollToAnalyzer();


    setTimeout(
        function() {

            analyzeTransaction();

        },
        700
    );
}


/* =========================================================
   PAGE INITIALIZATION
   ========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    function() {

        console.log(
            "Spider-Sense frontend initialized."
        );


        checkBackendHealth();
    }
);
