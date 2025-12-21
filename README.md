# From Claims to Charts  
### Visual Analytics of USPTO Patent Data (2024–2025)

**From Claims to Charts** is an end-to-end data analytics and visualization project that explores recent U.S. patenting activity using USPTO’s **PatentsView** dataset. The project combines large-scale data processing, statistical analysis, and interactive visualization to surface insights about innovation trends, technology domains, assignee behavior, and geographic patterns.

🔗 **Live Dashboard:**  
https://dashapp-349070754265.us-east1.run.app/

---

## Project Overview

This project analyzes **554,777 U.S. patents** granted between **January 2024 and June 2025**. Multiple relational tables from the PatentsView database were merged into a unified analytical dataset, enabling exploration across:
- Technology domains (CPC classifications)
- Patent structure (claims, figures, citations)
- Assignee type (corporate vs. academic)
- AI-related patents
- Geographic and regional patterns
The final outcome is an **interactive Dash dashboard** deployed on **Google Cloud Run**, designed for exploratory analysis of high-dimensional patent data.

---

## Why This Project Matters

Patent data is inherently complex: high-dimensional, skewed, and heterogeneous. This project demonstrates how **scalable data pipelines and visualization techniques** can transform raw legal-technical records into interpretable insights about innovation ecosystems.

Key takeaways include:
- Strong concentration of patenting activity in **electronics and computing**
- Distinct structural patterns between **academic and corporate patents**
- Clear **regional clustering** of technology specialization
- Heavy-tailed distributions in claims and citations, requiring robust statistical handling

---

## Data & Methodology

### Dataset Sources
- USPTO **PatentsView** relational tables (https://patentsview.org/download/data-download-tables)

### Data Processing
The preprocessing pipeline:
- Merges multiple PatentsView tables using patent IDs
- Normalizes categorical fields and resolves missing geographic data
- Engineers derived features such as:
  - `patent_complexity`
  - citations per claim
  - figures per claim
  - AI classification flags
- Applies **IQR-based outlier detection**
- Supports log, square-root, and **Box–Cox transformations**
- Uses **PCA** to explore variance structure in patent features

To keep the deployment lightweight, the dashboard uses a sample dataset of 50,000 patents.

All visuals are accessible through a **multi-tab Dash interface**.

---

## Tech Stack

- **Python**
- **Pandas, NumPy, SciPy**
- **Plotly & Dash**
- **Scikit-learn (PCA)**
- **Google Cloud Run**

---

## Author

**Mariana Soares**  
M.Eng. Computer Science & Applications — Virginia Tech  
Background in Computer Science, Data Analytics, and Technology Policy

This project was developed as part of **CS 5764 – Information Visualization**, but is designed and presented as a **production-quality data analytics portfolio project**.

