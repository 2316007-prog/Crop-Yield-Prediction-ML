/**
 * AgriYield AI - Interactive Front-End Script
 * Handles mobile navigation, tab switching, and client-side form validation.
 */

document.addEventListener("DOMContentLoaded", function () {
    // 1. Mobile Navigation Toggle
    const navToggle = document.getElementById("navToggle");
    const navLinks = document.getElementById("navLinks");

    if (navToggle && navLinks) {
        navToggle.addEventListener("click", function () {
            navLinks.classList.toggle("show");
        });
    }

    // 2. EDA Gallery Tab Switching
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    if (tabButtons.length > 0 && tabPanes.length > 0) {
        tabButtons.forEach(button => {
            button.addEventListener("click", function () {
                const targetTab = this.getAttribute("data-tab");

                // Deactivate all
                tabButtons.forEach(btn => btn.classList.remove("active"));
                tabPanes.forEach(pane => pane.classList.remove("active"));

                // Activate clicked
                this.classList.add("active");
                const activePane = document.getElementById(targetTab);
                if (activePane) {
                    activePane.classList.add("active");
                }
            });
        });
    }

    // 3. Prediction Form Validation
    const predictionForm = document.getElementById("predictionForm");
    const submitBtn = document.getElementById("submitBtn");

    if (predictionForm) {
        predictionForm.addEventListener("submit", function (e) {
            const crop = document.getElementById("crop");
            const state = document.getElementById("state");
            const area = document.getElementById("area");
            const rainfall = document.getElementById("rainfall");
            const fertilizer = document.getElementById("fertilizer");
            const temperature = document.getElementById("temperature");

            let isValid = true;
            let errorMessage = "";

            if (!crop || !crop.value) {
                isValid = false;
                errorMessage = "Please select a valid crop species.";
            } else if (!state || !state.value) {
                isValid = false;
                errorMessage = "Please select a state or region.";
            } else if (!area || parseFloat(area.value) <= 0 || isNaN(parseFloat(area.value))) {
                isValid = false;
                errorMessage = "Cultivated area must be a positive number greater than 0.";
            } else if (rainfall && (parseFloat(rainfall.value) < 0 || isNaN(parseFloat(rainfall.value)))) {
                isValid = false;
                errorMessage = "Annual rainfall cannot be negative.";
            } else if (fertilizer && (parseFloat(fertilizer.value) < 0 || isNaN(parseFloat(fertilizer.value)))) {
                isValid = false;
                errorMessage = "Fertilizer usage cannot be negative.";
            } else if (temperature && (parseFloat(temperature.value) < -20 || parseFloat(temperature.value) > 65)) {
                isValid = false;
                errorMessage = "Average temperature must be within a realistic meteorological range (-20°C to 65°C).";
            }

            if (!isValid) {
                e.preventDefault();
                alert(errorMessage);
                return false;
            }

            // Visual feedback on submission
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = `<span>⏳ Evaluating Prediction Model...</span>`;
            }
        });
    }

    // 4. Smooth Scroll to Prediction Results if available
    const activeResult = document.querySelector(".result-card.active-result");
    if (activeResult) {
        // Small delay to allow page rendering
        setTimeout(() => {
            activeResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }, 150);
    }
});
