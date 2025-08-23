# Nuclear-stain DNA-condensation metrics (CCP, CV, CI)

This note defines the **Chromatin Condensation Parameter (CCP)**, **Coefficient of Variation (CV)**, and a **Condensation Index (CI)** for per-nucleus computation. It specifies inputs, exact math, and normalization recommendations suitable for agentic implementation.

---

## Inputs & notation

- Grayscale nuclear channel image: \(I:\Omega\to\mathbb{R}_{\ge 0}\).
- Set of nuclei (segmented ROIs): \(\{N_1,\dots,N_K\}\), where \(N_n\subset\Omega\).
- For a nucleus \(n\):  
  - Pixel set \(P_n=\{p\in N_n\}\), size \(A_n=|P_n|\) (pixels).  
  - Per-nucleus intensity mean and std:
    \[
    \mu_n=\frac{1}{A_n}\sum_{p\in P_n} I(p),\qquad
    \sigma_n=\sqrt{\frac{1}{A_n}\sum_{p\in P_n}\big(I(p)-\mu_n\big)^2}.
    \]
- Percentile within a nucleus: \(Q_{n}(q)\) is the \(q\)-quantile of \(\{I(p):p\in P_n\}\). (E.g., \(Q_n(0.95)=P95\).)

### Per-nucleus intensity normalization (if used)
Define the z-scored, per-nucleus–normalized image:
\[
\hat I_n(p)=\frac{I(p)-\mu_n}{\sigma_n+\varepsilon},\quad p\in P_n,\quad \varepsilon>0\ \text{small (e.g., }10^{-8}\text{)}.
\]
Unless noted otherwise, formulas below use **raw \(I\)** (i.e., not \(\hat I_n\)).

---

## Summary: normalization recommendations

| Metric | Compute on per-nucleus normalized intensities? |
|---|---|
| **CV** | **No** (use raw \(I\)) |
| **CCP** | **No** (use raw \(I\); internal percentile threshold provides scale invariance) |
| **CI** | **No** (uses absolute upper-tail intensity vs. area; per-nucleus normalization would change its meaning) |

---

## 1) Coefficient of Variation (CV)

**Definition (per nucleus \(n\)):**
\[
\mathrm{CV}_n=\frac{\sigma_n}{\mu_n+\varepsilon}.
\]

**Implementation steps**
1. Gather \(\{I(p):p\in P_n\}\).  
2. Compute \(\mu_n,\sigma_n\).  
3. Return \(\mathrm{CV}_n=\sigma_n/(\mu_n+\varepsilon)\).

**Normalization recommendation:** **No.** CV already normalizes by \(\mu_n\) and is intended to reflect **within-nucleus dispersion relative to its mean**; per-nucleus z-scoring would collapse this contrast.

---

## 2) Chromatin Condensation Parameter (CCP)

**Goal:** quantify **intra-nuclear edge density** arising from sharp chromatin boundaries.

**Preliminaries**
- Use Sobel derivatives \(G_x= S_x * I,\ G_y= S_y * I\) with standard \(3\times3\) Sobel kernels \(S_x,S_y\).  
- Gradient magnitude \(G(p)=\sqrt{G_x(p)^2+G_y(p)^2}\).  
- To avoid border artifacts, optionally erode each nucleus by 1 px when forming the set used for gradient statistics (denote this eroded set \(P_n^\circ\subseteq P_n\)). (If you do not erode, use \(P_n\) for both steps.)

**Definition (per nucleus \(n\)):**
1. Compute \(G\) over the whole image once.  
2. Compute a **per-nucleus, percentile-based threshold** on gradient magnitude:
   \[
   \tau_n=\mathrm{quantile}\big(\{G(p):p\in P_n^\circ\},\ q\big),
   \]
   with a fixed \(q\in[0.80,0.95]\) (choose and keep constant; e.g., \(q=0.90\)).  
3. Edge-pixel set \(E_n=\{p\in P_n: G(p)\ge \tau_n\}\).  
4. **CCP** is the **edge fraction**:
   \[
   \mathrm{CCP}_n=\frac{|E_n|}{A_n}.
   \]

**Implementation steps**
1. Compute \(G\) (Sobel) once for the image.  
2. For each nucleus \(n\):  
   a. Form \(P_n^\circ\) (optional 1-px erosion).  
   b. Compute \(\tau_n\) as the \(q\)-quantile of \(\{G(p):p\in P_n^\circ\}\).  
   c. Count \(|E_n|=\sum_{p\in P_n} \mathbf{1}[G(p)\ge\tau_n]\).  
   d. Return \(\mathrm{CCP}_n=|E_n|/A_n\).

**Normalization recommendation:** **No.** Using a **per-nucleus gradient percentile threshold** \(\tau_n\) makes CCP scale-invariant; additional per-nucleus intensity normalization is unnecessary.

---

## 3) Condensation Index (CI)

**Goal:** combine **upper-tail nuclear intensity** and **nuclear area** into a single standardized score.

**Reference distributions (from control nuclei):**
- Collect a control set \(\mathcal{C}\) of nuclei.  
- Compute for controls:
  \[
  \mu_{P95}=\mathrm{mean}\big(\{Q_n(0.95): n\in\mathcal{C}\}\big),\quad
  \sigma_{P95}=\mathrm{std}\big(\{Q_n(0.95): n\in\mathcal{C}\}\big),
  \]
  \[
  \mu_{\log A}=\mathrm{mean}\big(\{\log A_n: n\in\mathcal{C}\}\big),\quad
  \sigma_{\log A}=\mathrm{std}\big(\{\log A_n: n\in\mathcal{C}\}\big).
  \]

**Definition (per nucleus \(n\)):**
\[
\mathrm{CI}_n
= \underbrace{\frac{Q_n(0.95)-\mu_{P95}}{\sigma_{P95}+\varepsilon}}_{\text{z-score of P95 intensity}}
\;-\;
\underbrace{\frac{\log A_n-\mu_{\log A}}{\sigma_{\log A}+\varepsilon}}_{\text{z-score of log-area (smaller }\Rightarrow\text{ larger term)}}.
\]

**Implementation steps**
1. Precompute \(\mu_{P95},\sigma_{P95},\mu_{\log A},\sigma_{\log A}\) from controls (once per experiment/batch).  
2. For each nucleus \(n\):  
   a. Compute \(P95_n=Q_n(0.95)\) from raw \(I\).  
   b. Compute \(A_n=|P_n|\) (or convert to \(\mu m^2\) if pixel size is known; the same units must be used for controls and samples).  
   c. Return \(\mathrm{CI}_n\) using the formula above.

**Normalization recommendation:** **No.** CI relies on **absolute upper-tail intensity** and **area**; per-nucleus normalization would remove the intended intensity contrast and alter CI’s interpretation.

---

## Implementation notes (common)

- Use a small numerical \(\varepsilon\) (e.g., \(10^{-8}\)) only to avoid division by zero.  
- Percentiles should use a deterministic interpolation method (e.g., “linear” or “nearest”) and be kept constant across runs.  
- All per-nucleus calculations must be restricted strictly to pixels \(p\in P_n\).
