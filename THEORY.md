# Multi-Band Triple-Filter Ensemble (MB-TFE): Theory, Motivation, and Expectations

> **Audience:** Students new to graph machine learning.  
> **Goal:** Explain the math behind our model, define the research question, and set realistic expectations.

---

## 1) Background: what are we doing and why?

**Graphs** (social networks, citation networks, molecules) connect objects (nodes) with relationships (edges). Each node has a **feature vector** (e.g., bag-of-words, attributes). In node classification we predict a label for each node.

A common observation:

- Some graphs are **homophilic**: neighbors tend to share the same label (e.g., papers cite papers in the same field).  
- Others are **heterophilic**: neighbors often have different labels (e.g., “question–answer” graphs).

A single “style” of message passing struggles to be good at both. Spectral methods fix this by **filtering** node signals at different **graph frequencies** (low = smooth across edges; high = varies rapidly across edges). The **TFE** idea (Triple-Filter Ensemble) mixes low- and high-frequency information (plus a skip of raw features) and works well across different homophily regimes.

**Our idea (MB-TFE):** extend the two extremes (low & high) into a small **bank of frequency bands**, obtained in a **stable** way from **differences of graph diffusions**. This makes “mid-frequency” content explicit, while keeping math, code, and compute simple.

---

## 2) Graphs, signals, and the graph Fourier view

- Graph \\(G=(V,E)\\), \\(|V|=n\\).  
- Adjacency \\(A\\in\\mathbb{R}^{n\\times n}\\) (symmetric for undirected graphs).  
- Degree \\(D=\\mathrm{diag}(A\\mathbf{1})\\).  
- **Normalized Laplacian**:
  \\[
  L_{\\mathrm{sym}} \\;=\\; I - D^{-1/2} A D^{-1/2} \\quad\\text{(eigenvalues in }[0,2]\\text{)}.
  \\]
- A **graph signal** is a matrix \\(X\\in\\mathbb{R}^{n\\times F}\\) (rows = nodes, columns = feature channels).

If \\(L_{\\mathrm{sym}} = U\\Lambda U^\\top\\) is the eigendecomposition, the **graph Fourier transform (GFT)** uses the orthonormal basis \\(U\\).  
- **Low eigenvalues** \\(\\lambda\\approx0\\) correspond to **smooth** patterns on the graph.  
- **High eigenvalues** \\(\\lambda\\approx 2\\) correspond to **rapidly varying** patterns.

A **spectral filter** is any function \\(h(\\lambda)\\) applied as \\(h(L_{\\mathrm{sym}})=U\\,h(\\Lambda)\\,U^\\top\\). We want to learn/use such filters **without** computing eigenvectors.

---

## 3) Heat-kernel diffusion = safe, low-pass smoothing

The **heat kernel** on graphs is
\\[
H_\\tau \\;=\\; e^{-\\tau L_{\\mathrm{sym}}},\\quad \\tau\\ge 0,
\\]
with spectral response \\(h_\\tau(\\lambda)=e^{-\\tau\\lambda}\\).  
- \\(\\tau=0\\) gives identity (no smoothing).  
- Larger \\(\\tau\\) means **stronger low-pass** (more smoothing).  
- It’s **stable** (positive, monotone, no sign flips).

We will **approximate** \\(H_\\tau X\\) by polynomials in \\(L_{\\mathrm{sym}}\\), which is efficient on large, sparse graphs.

---

## 4) Chebyshev polynomials: fast filtering without eigenvectors

Chebyshev polynomials \\(T_k(z)\\) satisfy \\(T_0=1,\\ T_1=z,\\ T_k=2zT_{k-1}-T_{k-2}\\) and are stable on \\([-1,1]\\).

Scale the Laplacian to this interval:
\\[
\\widehat{L} \\;=\\; \\tfrac{2}{\\lambda_{\\max}}L_{\\mathrm{sym}} - I \\ \\approx\\ L_{\\mathrm{sym}}-I
\\quad\\text{(since }\\lambda_{\\max}\\le 2\\text{)}.
\\]

Define the **Chebyshev basis** for features \\(X\\):
\\[
\\Psi_0=X,\\quad
\\Psi_1=\\widehat{L}X,\\quad
\\Psi_k=2\\widehat{L}\\Psi_{k-1}-\\Psi_{k-2} \\ (k\\ge2).
\\]
Each \\(\\Psi_k = T_k(\\widehat{L})X\\) can be computed by **sparse** matrix-vector products.

---

## 5) Closed-form Chebyshev coefficients for the heat kernel

We want the expansion \\(e^{-\\tau L_{\\mathrm{sym}}} \\approx \\sum_{k=0}^{K} a_k(\\tau)\\,T_k(\\widehat{L})\\).

Using the identity \\(e^{t z} = I_0(t) + 2\\sum_{k\\ge1} I_k(t)\\,T_k(z)\\) with modified Bessel functions \\(I_k\\),
\\[
e^{-\\tau\\lambda} \\;=\\; e^{-\\tau}\\, e^{-\\tau(\\lambda-1)} \\quad\\text{since } z=\\lambda-1 \\in [-1,1].
\\]
Thus the **Chebyshev coefficients** are:
\\[
\\boxed{
a_0(\\tau)=e^{-\\tau} I_0(\\tau),\\qquad
a_k(\\tau)=2\\,e^{-\\tau} (-1)^k I_k(\\tau)\\ \\ (k\\ge1).
}
\\]
Special case \\(\\tau=0\\): \\(a_0(0)=1,\\ a_{k>0}(0)=0\\).

Then the **smoothed features** are
\\[
Y(\\tau) \\;=\\; \\sum_{k=0}^{K} a_k(\\tau)\\,\\Psi_k \\;=\\; e^{-\\tau L_{\\mathrm{sym}}}X,
\\]
computed with only \\(K\\) sparse mat-vecs (to build \\(\\Psi_k\\)) plus cheap weighted sums.

---

## 6) Multi-band decomposition by **differences of diffusions**

Pick a small, increasing set of “temperatures” (band cut points):
\\[
0=\\tau_0 < \\tau_1 < \\cdots < \\tau_m.
\\]
Compute cumulative smoothings \\(Y(\\tau_i)\\) for \\(i=0,\\dots,m\\) (note \\(Y(\\tau_0)=X\\)).  
Define **bands** as successive differences:
\\[
\\boxed{
\\text{Band}_i \\;=\\; Y(\\tau_{i-1}) - Y(\\tau_i) \\quad (i=1,\\dots,m),\\qquad
\\text{UltraLow} \\;=\\; Y(\\tau_m).
}
\\]
Key properties:
- **Telescopes back to the input:** \\(\\sum_{i=1}^m \\text{Band}_i + \\text{UltraLow} = Y(\\tau_0) = X\\).  
- Uses **only low-pass operators** (stable); no sign ambiguities.  
- Provides explicit **mid-frequency** content (the differences).

---

## 7) The MB-TFE layer: full forward equations

Given \\(X \\in \\mathbb{R}^{n\\times F_{\\text{in}}}\\), \\(\\widehat{L}\\), order \\(K\\), and \\(\\{\\tau_i\\}_{i=1}^m\\):

1) **Chebyshev basis:** \\(\\Psi_0=X\\), \\(\\Psi_1=\\widehat{L}X\\), \\(\\Psi_k=2\\widehat{L}\\Psi_{k-1}-\\Psi_{k-2}\\).

2) **Bessel coefficients:** \\(a_0(\\tau)=e^{-\\tau}I_0(\\tau)\\), \\(a_k(\\tau)=2e^{-\\tau}(-1)^k I_k(\\tau)\\).

3) **Heat smoothings:** \\(Y(\\tau_i)=\\sum_{k=0}^K a_k(\\tau_i)\\Psi_k\\) for \\(i=0,\\dots,m\\) (with \\(\\tau_0=0\\)).

4) **Bands:**
   \\[
   \\text{Band}_i = Y(\\tau_{i-1})-Y(\\tau_i),\\quad i=1..m;\\qquad
   \\text{UltraLow}=Y(\\tau_m).
   \\]

5) **Per-band transforms (learned):**
   \\[
   H_i \\;=\\; \\sigma\\!\\big(\\text{Band}_i W_i + b_i\\big)\\ \\ (i=1..m),\\qquad
   H_0 \\;=\\; \\sigma\\!\\big(\\text{UltraLow} W_0 + b_0\\big),
   \\]
   with \\(W_i\\in\\mathbb{R}^{F_{\\text{in}}\\times F_b}\\).

6) **Fusion (TFE-style concat + skip):**
   \\[
   U \\;=\\; [\\,H_1\\ \\|\\ \\cdots\\ \\|\\ H_m\\ \\|\\ H_0\\ \\|\\ X\\,],\\qquad
   Z \\;=\\; U\\,W_{\\mathrm{fuse}} + b_{\\mathrm{fuse}},
   \\]
   where \\(W_{\\mathrm{fuse}}\\in\\mathbb{R}^{((m+1)F_b+F_{\\text{in}})\\times F_{\\text{out}}}\\).

7) **Optional diversity regularizer** (encourage non-redundant bands):
   \\[
   \\widetilde{H}_i[u,:] \\;=\\; \\frac{H_i[u,:]}{\\|H_i[u,:]\\|_2+\\varepsilon},\\quad
   \\mathcal{L}_{\\text{div}} \\;=\\; \\lambda_{\\text{div}} \\sum_{i<j}\\big\\|\\widetilde{H}_i^\\top \\widetilde{H}_j\\big\\|_F^2.
   \\]

> **Complexity:** computing \\(\\{\\Psi_k\\}\\) costs \\(K\\) sparse mat-vecs → \\(O(K\\,|E|\\,F_{\\text{in}})\\). Forming \\(Y(\\tau_i)\\) and bands is just weighted sums of \\(\\Psi_k\\) → cheap. Dense per-band maps and fusion are \\(O(n\\cdot\\text{dims})\\).

---

## 8) Relation to (and fixes over) classic TFE

**Classic TFE** mixes **low-pass** and **high-pass** plus skip. The “HP” is effectively a complement (e.g., \\(X - \\text{LP}_\\tau\\)), which is fine, but **mid-frequency** content is not explicit.

**MB-TFE**:
- Decomposes \\(X\\) into **multiple bands** via **differences of heat smoothings** (still stable).  
- Keeps fusion as **concat + MLP** (TFE’s robust choice).  
- Uses a **single operator** \\(L_{\\mathrm{sym}}\\) and stable scaling \\(\\widehat{L}=L_{\\mathrm{sym}}-I\\) → no sign confusion or mixed conventions.  
- Adds a small **diversity loss** to avoid band collapse.

---

## 9) Research question & hypotheses

### Research question
> **Does explicitly modeling mid-frequency information via stable “difference-of-diffusions” bands improve node-classification accuracy and robustness over standard TFE and common baselines, especially on graphs with mixed homophily?**

### Hypotheses
1. **H1 (accuracy):** On **mixed-homophily** datasets, MB-TFE achieves **higher test accuracy** than TFE (and similar compute).  
2. **H2 (robustness):** MB-TFE degrades **no faster** than TFE under feature noise or random edge dropout.  
3. **H3 (parsimony):** With small \\(K\\) (e.g., 5–10) and a few bands (e.g., 2–4), MB-TFE matches TFE’s training time **within ~10–20%**.  
4. **H4 (diagnostics):** Learned bands show **non-redundant** energies (with diversity loss) and align with the dataset’s label smoothness scale.

---

## 10) What outcomes to expect (realistically)

- **Where MB-TFE helps most:** graphs that are neither very homophilic nor very heterophilic (e.g., web/wikipedia/citation graphs with mixed structure). Expect **modest but consistent** gains over TFE (tenths to low single-digit %).  
- **Where gains are small:** very homophilic or very heterophilic graphs—two bands already capture most signal.  
- **Compute:** essentially the same **sparse** cost as TFE at the same \\(K\\); a few more **dense** ops in fusion due to concatenation of extra bands.  
- **Stability:** difference-of-diffusions avoids sign flips and remains numerically well-behaved.

---

## 11) Practical defaults (so you can run today)

- **Bands:** \\(m=3\\), \\(\\tau=\\{0.5,\\,1.5,\\,4.0\\}\\) (log-spaced).  
- **Order:** \\(K=8\\).  
- **Dimensions:** \\(F_b=F_{\\text{in}}\\); hidden size \\(=128\\).  
- **Regularization:** dropout \\(0.5\\), weight decay \\(5\\cdot10^{-5}\\), \\(\\lambda_{\\text{div}}=10^{-3}\\).  
- **Fusion:** concat + linear + ReLU.

---

## 12) Limitations & edge cases

- If the graph has **extreme** homophily (near 1) or **extreme** heterophily (near 0), extra bands may not help.  
- If \\(K\\) is too large and the model is over-parameterized, **overfitting** can happen—watch validation curves; use early stopping and regularization.  
- Very sparse graphs with many isolated nodes can reduce the usefulness of diffusion; we handle degrees with a safe floor in normalization, but performance may be bounded by data quality.

---

## 13) Glossary of symbols

- \\(G=(V,E)\\): graph; \\(n=|V|\\) nodes.  
- \\(A\\): adjacency; \\(D=\\mathrm{diag}(A\\mathbf{1})\\).  
- \\(L_{\\mathrm{sym}}=I-D^{-1/2}AD^{-1/2}\\): normalized Laplacian.  
- \\(\\widehat{L}=L_{\\mathrm{sym}}-I\\): scaled Laplacian for Chebyshev on \\([-1,1]\\).  
- \\(X\\in\\mathbb{R}^{n\\times F}\\): node features.  
- \\(T_k\\): Chebyshev polynomials.  
- \\(\\Psi_k=T_k(\\widehat{L})X\\): Chebyshev basis features.  
- \\(H_\\tau=e^{-\\tau L_{\\mathrm{sym}}}\\): heat diffusion (low-pass).  
- \\(a_k(\\tau)\\): Chebyshev coefficients of \\(H_\\tau\\) (use Bessel \\(I_k\\)).  
- \\(Y(\\tau)\\): smoothed features at temperature \\(\\tau\\).  
- \\(\\text{Band}_i=Y(\\tau_{i-1})-Y(\\tau_i)\\): band features.  
- \\(H_i=\\sigma(\\text{Band}_iW_i+b_i)\\): per-band transform.  
- \\(Z=[H_1\\|\\dots\\|H_m\\|H_0\\|X]W_{\\mathrm{fuse}}+b_{\\mathrm{fuse}}\\): fused output.  
- \\(\\mathcal{L}_{\\text{div}}\\): diversity regularizer.

---

## 14) One-page summary (for your README)

- Build a Chebyshev basis once, approximate heat kernels at a few \\(\\tau\\)’s with closed-form Bessel coefficients.  
- Form **bands** as differences of successive smoothings (stable, interpretable).  
- Apply tiny per-band linear maps, **concatenate** bands + raw skip, and classify.  
- Expect wins mainly on **mixed-homophily** graphs, with near-TFE compute.
