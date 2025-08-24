document.addEventListener("DOMContentLoaded", function () {
  // Default chart display
  showChart("pieChartSection");

  // Radar chart: populate student selector
  if (typeof studentNames !== "undefined") {
    const selector = document.getElementById("studentSelector");
    studentNames.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.text = name;
      selector.appendChild(option);
    });
    selector.addEventListener("change", function () {
      updateRadarChart(this.value);
    });
  }

  // Render Charts
  if (typeof Chart !== "undefined") {
    renderPieChart();
    renderBarChart();
    renderLineChart();
    if (studentNames && studentNames.length > 0) {
      updateRadarChart(studentNames[0]);
    }
  }

  // -------- Risk filter logic --------
  const riskFilter = document.getElementById("riskFilter");
  if (riskFilter) {
    function getRiskColumnIndex(table) {
      if (!table) return -1;
      const headers = table.querySelectorAll("thead th");
      for (let i = 0; i < headers.length; i++) {
        const txt = (headers[i].textContent || "").trim().toLowerCase();
        const norm = txt.replace(/[^a-z0-9]/g, "");
        if (norm.includes("risk")) return i;
      }
      return -1;
    }

    riskFilter.addEventListener("change", function () {
      const selected = (this.value || "").trim().toLowerCase();
      const table = document.querySelector("table.dataframe") || document.querySelector("table");
      if (!table) return;

      const riskColIndex = getRiskColumnIndex(table);
      if (riskColIndex === -1) return;

      const rows = table.querySelectorAll("tbody tr");
      rows.forEach(row => {
        const cells = row.querySelectorAll("td");
        if (!cells || cells.length <= riskColIndex) {
          row.style.display = "";
          return;
        }
        const cellText = (cells[riskColIndex].textContent || "").trim().toLowerCase();
        row.style.display = (selected === "all" || cellText === selected) ? "" : "none";
      });
    });
  }

  // -------- Focus Areas Feature (Updated Styling + Visualization) --------
  const focusSelector = document.getElementById("focusStudentSelector");
  if (focusSelector) {
    focusSelector.addEventListener("change", function () {
      const selectedStudent = this.value;
      const listDiv = document.getElementById("focusList");
      listDiv.innerHTML = ''; // clear previous

      if (!selectedStudent || !studentFocusAreas[selectedStudent]) {
        listDiv.innerHTML = "<p>Please select a student to see focus areas.</p>";
        return;
      }

      const areas = studentFocusAreas[selectedStudent];
      if (areas.length === 0) {
        listDiv.innerHTML = `<p class="text-success">${selectedStudent} is performing well in all subjects and attendance.</p>`;
      } else {
        areas.forEach(item => {
          // Card container
          const card = document.createElement("div");
          card.classList.add("mb-3", "p-3", "border", "rounded", "shadow-sm");
          card.style.backgroundColor = "#f8f9fa";

          // Title
          const title = document.createElement("h6");
          title.textContent = `Focus on: ${item}`;
          title.classList.add("fw-bold", "mb-2");
          title.style.color = "#dc3545";

          // Progress Bar (Random level for now)
          const progressContainer = document.createElement("div");
          progressContainer.classList.add("progress");
          const progressBar = document.createElement("div");
          progressBar.classList.add("progress-bar", "bg-danger");
          progressBar.style.width = Math.floor(Math.random() * 40 + 40) + "%";
          progressBar.textContent = "Needs Improvement";

          progressContainer.appendChild(progressBar);

          // Append
          card.appendChild(title);
          card.appendChild(progressContainer);
          listDiv.appendChild(card);
        });
      }
    });
  }

  // Sidebar toggle for icon-only view
  const sidebarToggle = document.getElementById("sidebarToggle");
  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", function () {
      document.getElementById("sidebar").classList.toggle("collapsed");
    });
  }
});

// ---------------- Helper Functions ----------------
function showChart(sectionId) {
  const sections = ["pieChartSection", "barChartSection", "trendChartSection", "radarChartSection"];
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = (id === sectionId) ? "block" : "none";
  });

  // Hide focus area section when showing any chart
  const focusSection = document.getElementById('focusAreaSection');
  if (focusSection) focusSection.style.display = 'none';
}

function renderPieChart() {
  new Chart(document.getElementById("pieChart"), {
    type: "pie",
    data: {
      labels: Object.keys(riskData),
      datasets: [{
        label: "Students",
        data: Object.values(riskData),
        backgroundColor: ["#28a745", "#ffc107", "#dc3545"]
      }]
    },
    options: { responsive: true, plugins: { title: { display: true, text: "Risk Level Distribution" } } }
  });
}

function renderBarChart() {
  new Chart(document.getElementById("barChart"), {
    type: "bar",
    data: {
      labels: Object.keys(subjectData),
      datasets: [{ label: "Average Marks", data: Object.values(subjectData), backgroundColor: "#007bff" }]
    },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: "Average Subject Marks" } },
      scales: { y: { beginAtZero: true, max: 100 } }
    }
  });
}

function renderLineChart() {
  new Chart(document.getElementById("trendChart"), {
    type: "line",
    data: {
      labels: semesters,
      datasets: Object.keys(trendData).map(subject => ({
        label: subject,
        data: trendData[subject],
        fill: false,
        tension: 0.1
      }))
    },
    options: { responsive: true, plugins: { title: { display: true, text: "Subject Trends Over Semesters" } } }
  });
}

function updateRadarChart(studentName) {
  const ctx = document.getElementById("radarChart");
  if (ctx.radarInstance) ctx.radarInstance.destroy();

  const subjectScores = studentRadarData[studentName];
  ctx.radarInstance = new Chart(ctx, {
    type: "radar",
    data: {
      labels: Object.keys(subjectScores),
      datasets: [{
        label: studentName,
        data: Object.values(subjectScores),
        fill: true,
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        borderColor: "rgba(54, 162, 235, 1)"
      }]
    },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: "Subject-wise Performance" } },
      scales: { r: { beginAtZero: true, max: 100 } }
    }
  });
}

function showFocusAreaSection() {
  const section = document.getElementById('focusAreaSection');
  if (section) {
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth' });
  }

  // Hide all chart sections when showing focus area
  const chartSections = ["pieChartSection", "barChartSection", "trendChartSection", "radarChartSection"];
  chartSections.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
}
