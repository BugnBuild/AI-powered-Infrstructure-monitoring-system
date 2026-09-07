// ============================================
// SMART CIVIC AI — script.js
// ============================================

const API_URL = "http://127.0.0.1:8000";

let currentLatitude  = 23.2599;
let currentLongitude = 77.4126;
let currentLocality  = "Unknown";   // reverse-geocoded locality name


// ============================================
// GPS — REVERSE GEOCODE HELPER
// ============================================

/**
 * Uses the free OpenStreetMap Nominatim API to convert
 * lat/lon into a human-readable locality string.
 * e.g. "Arera Colony, Bhopal, Madhya Pradesh"
 */
async function reverseGeocode(lat, lon) {
    try {
        const url =
            `https://nominatim.openstreetmap.org/reverse` +
            `?lat=${lat}&lon=${lon}&format=json&zoom=14&accept-language=en`;

        const resp = await fetch(url, {
            headers: { "User-Agent": "SmartCivicAI/1.0" }
        });
        if (!resp.ok) throw new Error("Nominatim HTTP " + resp.status);

        const data = await resp.json();
        const a    = data.address || {};

        // Build locality: neighbourhood → city → state (max 3 parts)
        const parts = [
            a.suburb || a.neighbourhood || a.village || a.town || a.city_district || "",
            a.city   || a.town          || a.county  || "",
            a.state  || ""
        ].map(s => s.trim()).filter(Boolean);

        // Remove consecutive duplicates
        const unique = parts.filter((v, i) => v !== parts[i - 1]);
        return unique.slice(0, 3).join(", ") || data.display_name || "Unknown";

    } catch (err) {
        console.warn("Reverse geocode error:", err);
        return "Unknown";
    }
}


// ============================================
// GPS — DETECT WITH PERMISSION REQUEST
// ============================================

async function getLocation() {

    const text   = document.getElementById("locationText");
    const button = document.querySelector(".location-btn");
    const pin    = document.querySelector(".location-icon");

    // ── No browser support ──────────────────────────────────────────────────
    if (!navigator.geolocation) {
        text.innerHTML =
            `<span class="loc-error">⚠️ GPS not supported by this browser.</span>`;
        return;
    }

    // ── Show requesting state ───────────────────────────────────────────────
    text.innerHTML =
        `<span class="loc-status">` +
            `<span class="loc-spinner">⏳</span> Requesting location permission…` +
        `</span>`;

    if (button) { button.disabled = true; button.innerText = "Detecting…"; }
    if (pin)    { pin.textContent = "⏳"; }

    // ── Request GPS ─────────────────────────────────────────────────────────
    navigator.geolocation.getCurrentPosition(

        // SUCCESS ─────────────────────────────────────────────────────────────
        async function(position) {

            currentLatitude  = position.coords.latitude;
            currentLongitude = position.coords.longitude;

            // Immediately show coords while geocoding runs
            text.innerHTML =
                `<span class="loc-status">` +
                    `<span class="loc-spinner">🌐</span> Identifying locality…` +
                `</span>`;

            // Reverse geocode in background
            const locality = await reverseGeocode(currentLatitude, currentLongitude);
            currentLocality = locality;

            // Update the location box with full details
            text.innerHTML =
                `<span class="loc-locality">📍 ${escapeHtml(locality)}</span>` +
                `<span class="loc-coords">` +
                    `${currentLatitude.toFixed(6)}, ${currentLongitude.toFixed(6)}` +
                `</span>`;

            if (button) {
                button.disabled  = false;
                button.innerText = "✓ Detected";
                button.classList.add("detected");
            }
            if (pin) pin.textContent = "📍";
        },

        // ERROR / DENIED ──────────────────────────────────────────────────────
        function(error) {

            console.warn("GPS error:", error.code, error.message);

            let msg;
            switch (error.code) {
                case error.PERMISSION_DENIED:
                    msg = "Location access denied. Allow location in browser settings and retry.";
                    break;
                case error.POSITION_UNAVAILABLE:
                    msg = "Location signal unavailable. Demo coordinates will be used.";
                    break;
                case error.TIMEOUT:
                    msg = "Location request timed out. Please retry.";
                    break;
                default:
                    msg = "GPS unavailable. Demo coordinates will be used.";
            }

            currentLocality = "Unknown";
            text.innerHTML  =
                `<span class="loc-error">⚠️ ${msg}</span>` +
                `<span class="loc-coords">Demo: ` +
                    `${currentLatitude.toFixed(6)}, ${currentLongitude.toFixed(6)}` +
                `</span>`;

            if (button) {
                button.disabled  = false;
                button.innerText = "Retry";
                button.classList.remove("detected");
            }
            if (pin) pin.textContent = "📍";
        },

        // Options ─────────────────────────────────────────────────────────────
        { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
    );
}


// ============================================
// OPEN MODAL — auto-detect GPS on open
// ============================================

function openReport() {
    const modal = document.getElementById("reportModal");
    if (modal) modal.classList.add("show");

    // Auto-request GPS 400ms after modal opens (lets animation finish)
    setTimeout(() => {
        const text = document.getElementById("locationText");
        if (text && text.textContent.trim().startsWith("Location will")) {
            getLocation();
        }
    }, 400);
}


// ============================================
// CLOSE MODAL
// ============================================

function closeReport() {
    const modal = document.getElementById("reportModal");
    if (modal) modal.classList.remove("show");
}


// ============================================
// IMAGE PREVIEW
// ============================================

function previewImage(event) {
    const file = event.target.files[0];
    if (!file) return;

    const allowed = ["image/jpeg", "image/png", "image/webp"];
    if (!allowed.includes(file.type)) {
        alert("Please select a JPG, PNG or WEBP image.");
        event.target.value = "";
        return;
    }

    const preview = document.getElementById("preview");
    preview.src   = URL.createObjectURL(file);
    preview.style.display = "block";
}


// ============================================
// SUBMIT BUTTON
// ============================================

function submitDemo() {
    const file = document.getElementById("imageInput").files[0];
    if (!file) {
        alert("Please upload a road image first.");
        return;
    }
    showSubmitConfirmation();
}


// ============================================
// CONFIRMATION POPUP
// ============================================

function showSubmitConfirmation() {
    closeSubmitConfirmation();

    const popup = document.createElement("div");
    popup.id    = "submitConfirmation";

    const localityLine = currentLocality !== "Unknown"
        ? `<p class="confirm-locality">📍 ${escapeHtml(currentLocality)}</p>`
        : "";

    popup.innerHTML = `
        <div class="confirmation-card">
            <button class="confirmation-close" onclick="closeSubmitConfirmation()" type="button">×</button>
            <div class="confirmation-icon">?</div>
            <h2>Submit Complaint?</h2>
            ${localityLine}
            <p>Your road image will be analyzed by Smart Civic AI and submitted as a civic complaint.</p>
            <div class="confirmation-actions">
                <button class="cancel-submit"  onclick="closeSubmitConfirmation()"      type="button">Cancel</button>
                <button class="confirm-submit" onclick="confirmComplaintSubmission()"   type="button">Confirm Submit</button>
            </div>
        </div>`;

    document.body.appendChild(popup);
    popup.addEventListener("click", e => { if (e.target === popup) closeSubmitConfirmation(); });
}

function closeSubmitConfirmation() {
    const p = document.getElementById("submitConfirmation");
    if (p) p.remove();
}


// ============================================
// SUBMIT TO BACKEND
// ============================================

async function confirmComplaintSubmission() {
    closeSubmitConfirmation();

    const file = document.getElementById("imageInput").files[0];
    if (!file) { alert("Please upload a road image first."); return; }

    const formData = new FormData();
    formData.append("user_id",   "1");
    formData.append("latitude",  currentLatitude);
    formData.append("longitude", currentLongitude);
    formData.append("locality",  currentLocality);   // ← send locality to backend
    formData.append("file",      file);

    const button = document.querySelector(".submit-btn");

    try {
        button.disabled  = true;
        button.innerText = "AI Analyzing…";

        const response = await fetch(`${API_URL}/complaints/submit`, {
            method: "POST",
            body: formData
        });

        const responseText = await response.text();
        console.log("Backend response:", responseText);

        let result = null;
        try { result = JSON.parse(responseText); } catch (_) {}

        if (!response.ok) {
            const message = result?.detail || "Complaint submission failed.";
            alert("Could not submit complaint.\n\n" + message);
            return;
        }

        // Success
        const analysis    = result?.analysis || {};
        const damageType  = analysis.damage_type || "Unknown";
        const confidence  = Number(analysis.confidence || 0);
        const severity    = analysis.severity    || "Unknown";
        const complaintId = result?.complaint_id || "N/A";
        const datasetPath = result?.dataset_path || "";
        const locality    = result?.location?.locality || currentLocality;

        closeReport();
        resetReportForm();

        let msg =
            "✅ Complaint Submitted!\n\n" +
            "Damage Type : " + damageType + "\n" +
            "Confidence  : " + (confidence * 100).toFixed(1) + "%\n" +
            "Severity    : " + severity + "\n" +
            "Location    : " + (locality !== "Unknown" ? locality : `${currentLatitude.toFixed(4)}, ${currentLongitude.toFixed(4)}`) + "\n" +
            "Complaint # : #" + complaintId;

        if (datasetPath && datasetPath !== "skipped") {
            msg += "\n\n📁 Image saved to dataset: " + datasetPath;
        }

        alert(msg);
        loadComplaints();

    } catch (error) {
        console.error("Fetch error:", error);
        loadComplaints();
        alert("The complaint may already have been saved.\nPlease check Recent Complaints.");
    } finally {
        button.disabled  = false;
        button.innerText = "Submit Complaint";
    }
}


// ============================================
// RESET FORM
// ============================================

function resetReportForm() {
    const input    = document.getElementById("imageInput");
    const preview  = document.getElementById("preview");
    const locText  = document.getElementById("locationText");
    const button   = document.querySelector(".location-btn");

    if (input)   input.value = "";
    if (preview) { preview.src = ""; preview.style.display = "none"; }
    if (locText) locText.innerHTML = "Location will be detected automatically";
    if (button)  { button.innerText = "Detect"; button.classList.remove("detected"); }

    currentLatitude  = 23.2599;
    currentLongitude = 77.4126;
    currentLocality  = "Unknown";
}


// ============================================
// LOAD COMPLAINTS
// ============================================

async function loadComplaints() {
    try {
        const resp = await fetch(`${API_URL}/complaints/`);
        if (!resp.ok) throw new Error("Load failed");
        const complaints = await resp.json();
        updateComplaintTable(complaints);
        updateStatistics(complaints);
    } catch (err) {
        console.error("Load complaints error:", err);
    }
}


// ============================================
// IMAGE URL HELPER
// ============================================

function getComplaintImageUrl(imagePath) {
    if (!imagePath) return "";
    const filename = imagePath.replace(/\\/g, "/").split("/").pop();
    return `${API_URL}/uploads/${encodeURIComponent(filename)}`;
}


// ============================================
// UPDATE TABLE
// ============================================

function updateComplaintTable(complaints) {
    const table  = document.querySelector(".complaint-table");
    if (!table) return;

    const header = table.querySelector(".table-header");
    table.innerHTML = "";
    if (header) table.appendChild(header);

    if (!complaints || complaints.length === 0) {
        const row = document.createElement("div");
        row.className = "table-row";
        row.innerHTML = `
            <span>—</span>
            <span class="issue">
                <div class="issue-icon">📋</div>
                <div class="issue-details"><strong>No complaints yet</strong><small>Submit a civic issue</small></div>
            </span>
            <span>—</span><span>—</span><span>—</span>`;
        table.appendChild(row);
        return;
    }

    [...complaints]
        .sort((a, b) => Number(b.id) - Number(a.id))
        .forEach(complaint => {
            const row      = document.createElement("div");
            row.className  = "table-row";

            const damage   = complaint.damage_type || "Unknown";
            const severity = complaint.severity    || "Unknown";
            const status   = complaint.status      || "Submitted";
            const imageUrl = getComplaintImageUrl(complaint.image_path);

            // Location: show locality name if available, else coords
            const locality = complaint.locality && complaint.locality !== "Unknown"
                ? escapeHtml(complaint.locality)
                : `${Number(complaint.latitude).toFixed(4)}, ${Number(complaint.longitude).toFixed(4)}`;

            const imageHTML = imageUrl
                ? `<img src="${imageUrl}" class="complaint-image" alt="Road damage">`
                : `<div class="issue-icon">${getIssueIcon(damage)}</div>`;

            row.innerHTML = `
                <span>#${complaint.id}</span>
                <span class="issue">
                    ${imageHTML}
                    <div class="issue-details">
                        <strong>${escapeHtml(damage)}</strong>
                        <small>Road damage</small>
                    </div>
                </span>
                <span><b class="severity ${getSeverityClass(severity)}">${escapeHtml(severity)}</b></span>
                <span title="${Number(complaint.latitude).toFixed(6)}, ${Number(complaint.longitude).toFixed(6)}">
                    📍 ${locality}
                </span>
                <span><b class="status ${getStatusClass(status)}">${escapeHtml(status)}</b></span>`;

            table.appendChild(row);
        });
}


// ============================================
// STATISTICS
// ============================================

function updateStatistics(complaints) {
    const cards = document.querySelectorAll(".stat-card");
    if (cards.length < 3) return;

    const total      = complaints.length;
    const resolved   = complaints.filter(c => String(c.status || "").toLowerCase() === "resolved").length;
    const underReview = total - resolved;

    cards[0].querySelector("strong").innerText = total;
    cards[1].querySelector("strong").innerText = underReview;
    cards[2].querySelector("strong").innerText = resolved;
}


// ============================================
// HELPERS
// ============================================

function getSeverityClass(severity) {
    const v = String(severity || "").toLowerCase();
    if (v === "high") return "high";
    if (v === "low")  return "low";
    return "medium";
}

function getStatusClass(status) {
    const v = String(status || "").toLowerCase();
    if (v === "resolved") return "resolved";
    if (v === "reviewing" || v === "under review") return "reviewing";
    return "submitted";
}

function getIssueIcon(damage) {
    const v = String(damage || "").toLowerCase();
    if (v.includes("pothole")) return "🕳️";
    if (v.includes("crack"))   return "〰️";
    return "🛣️";
}

function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = String(value);
    return div.innerHTML;
}


// ============================================
// MODAL — OUTSIDE CLICK & ESC
// ============================================

const reportModal = document.getElementById("reportModal");
if (reportModal) {
    reportModal.addEventListener("click", function(e) {
        if (e.target === this) closeReport();
    });
}

document.addEventListener("keydown", function(e) {
    if (e.key === "Escape") { closeReport(); closeSubmitConfirmation(); }
});


// ============================================
// INITIAL LOAD
// ============================================

document.addEventListener("DOMContentLoaded", function() {
    loadComplaints();
});
