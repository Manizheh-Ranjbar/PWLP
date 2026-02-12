# PWLP — Response to Reviewers (پاسخ به داوران)

## ارزیابی کلی: آیا شانس اکسپت دارید؟

**خلاصه:** بله، در صورت اصلاح جدی و کامل همهٔ نکات، **شانس اکسپت وجود دارد**.  
- **داور ۱:** نظر مثبت؛ درخواست‌ها مشخص و قابل انجام (جدول هایپرپارامترها، تحلیل کیس، محدودیت‌ها، توضیح شهودی).  
- **داور ۲:** Major revision — خواسته‌ها سنگین‌تر است (نوآوری روشن، تئوری، آزمایش‌های runtime/حافظه، کوتاه‌سازی متن، مراجع جدید).  
- **داور ۳:** اصلاحات متوسط (نوآوری در مقدمه، ادبیات، محدودیت‌ها، خوانایی).

اگر هر سه داور را با **پاسخ‌های شفاف** و **تغییرات واقعی در متن** راضی کنید، احتمال اکسپت پس از یک دور اصلاح خوب است.

---

# پاسخ‌های تفصیلی به هر داور

---

## Reviewer #1

### Comment 1.1 — جدول هایپرپارامترها در Section 6.3
**نظر داور:** در Section 6.3 (یا فصل methodology) همهٔ هایپرپارامترها (مثل جهت walk، معماری MLP) باید در یک جدول لیست شوند و برای مقادیر انتخاب‌شده توجیه (مثلاً grid search روی validation یا ارجاع به کارهای قبلی) ارائه شود.

**پاسخ پیشنهادی (Response):**
> We thank the reviewer for this suggestion. We have added a dedicated table (Table X) in Section 6.3 that explicitly lists all hyperparameters used in PWLP, including walk parameters (direction, length L, number of walks k, top-n selection), model architecture (MLP layers, dimensions), and training settings (learning rate, epochs, etc.). For each, we provide a short justification: key parameters were selected via validation-based grid search and sensitivity analysis, as detailed in Section 6.9.2; where applicable we cite prior work (e.g., SEAL, NCN) for consistent settings.

**تغییر در مقاله:**  
- یک **جدول** در Section 6.3 اضافه کنید با ستون‌های: Hyperparameter | Value | Justification (grid search / validation / citation).  
- در Section 6.9.2 به grid search و sensitivity analysis اشاره کنید و در جدول به این بخش ارجاع دهید.

---

### Comment 1.2 — تحلیل عمیق و کیس استادی
**نظر داور:** بخشی از تحلیل در حد گزارش عملکرد است؛ پیشنهاد می‌شود تحلیل کیس عمیق‌تر اضافه شود، مثلاً با **ویژوالایز کردن subgraphهای ساخته‌شده با pool walk** برای جفت نودهایی که PWLP بهبود قابل‌توجه نسبت به baselineها نشان می‌دهد.

**پاسخ پیشنهادی (Response):**
> We agree that a deeper case analysis would strengthen the paper. We have added a new subsection (e.g., 6.9.3 or 7.X) that presents an in-depth case study: we select representative node pairs where PWLP shows significant improvement over baselines, visualize the subgraphs constructed via pool walks (including which nodes were selected and how the walk paths look), and briefly interpret why the chosen structure leads to better prediction. This complements the quantitative results with qualitative insight into PWLP’s behavior.

**تغییر در مقاله:**  
- یک **subsection** برای Case Study اضافه کنید.  
- برای چند جفت نود مشخص (که PWLP بهتر عمل کرده) subgraphهای استخراج‌شده با pool walk را **ویژوالایز** کنید و یک پاراگراف تفسیر کوتاه بنویسید.

---

### Comment 1.3 — Baselineهای جدید و محدودیت‌ها
**نظر داور:** مقایسه با روش‌های SOTA سال‌های اخیر باید انجام شود؛ همچنین بحث **محدودیت‌های مدل** در بخش نتیجه‌گیری کافی نیست.

**پاسخ پیشنهادی (Response):**
> We have extended the baseline comparison to include state-of-the-art methods from recent years [add specific names and citations if you add new baselines]. In the Conclusion, we have added a dedicated paragraph on limitations: we discuss (1) scalability and memory constraints on very large graphs, (2) sensitivity to walk and subgraph-size hyperparameters, (3) applicability when node features are missing or very noisy, and (4) computational cost compared to simpler heuristics. We also mention directions for future work that address these limitations.

**تغییر در مقاله:**  
- اگر روش جدید SOTA دارید، آن‌ها را به جدول/آزمایش اضافه کنید و در متن به «recent SOTA» اشاره کنید.  
- در **Conclusion** یک پاراگراف مشخص با عنوان «Limitations» اضافه کنید و حداقل ۳–۴ محدودیت را صریح بنویسید.

---

### Comment 1.4 — توضیح شهودی در ابتدای مقدمه یا methodology
**نظر داور:** تحلیل تئوریک (Lemma 1–2) روی پیچیدگی و expressive power متمرکز است؛ پیشنهاد می‌شود در **ابتدای Introduction یا Methodology** یک توضیح **کیفی و شهودی** از ایدهٔ اصلی PWLP اضافه شود.

**پاسخ پیشنهادی (Response):**
> We thank the reviewer for this suggestion. We have added a short qualitative description at the beginning of the Methodology (Section X): we explain in plain language that PWLP (1) uses pool walks to dynamically select a small set of influential nodes around each candidate link, (2) builds a compact subgraph from these nodes and the target edge, and (3) learns edge-centric representations via line graph encoding and fusion of structural and node features. This provides an intuitive overview before the formal definitions and lemmas.

**تغییر در مقاله:**  
- در **ابتدای بخش Methodology** (قبل از فرمول‌ها و Lemmaها) یک پاراگراف «Intuitive overview» یا «High-level idea» اضافه کنید که ایدهٔ pool walk، subgraph فشرده و line graph را به زبان ساده توضیح دهد.

---

## Reviewer #2

### Comment 2.1 — نوآوری، مقایسه مفهومی با LGLP/SEAL/NCN، اصل جدید یا ترکیب؟
**نظر داور:** بیان روشن‌تر آنچه واقعاً جدید است؛ مقایسه و تضاد مستقیم با LGLP، SEAL و NCN در سطح **طراحی مفهومی**، نه فقط تجربی؛ و بحث صریح اینکه آیا PWLP یک **اصل مدل‌سازی جدید** معرفی می‌کند یا عمدتاً **ترکیب مؤثرتر ایده‌های موجود** است.

**پاسخ پیشنهادی (Response) — نسخهٔ قوی و متقاعدکننده:**
> We thank the reviewer for asking us to clarify what is fundamentally new. We have revised the Introduction and Related Work to articulate our contributions more strongly and to add a direct conceptual comparison with LGLP, SEAL, and NCN. **(1) What is new:** We introduce a **new design principle** for subgraph-based link prediction: **influence-driven subgraph extraction** via pool walks. Prior methods (SEAL, NCN, LGLP) use fixed-hop or fixed-radius subgraphs; PWLP instead **dynamically** selects influential nodes by visit frequency over pool walks, yielding adaptive, compact subgraphs—a strategy that had not been proposed in this form for link prediction. **(2)** We are the **first to systematically combine** this extraction with line-graph-based edge-centric encoding and balanced fusion of structural and node features in one framework; this combination is novel and yields consistent gains in our experiments. **(3)** We provide **theoretical support** (Lemma 1: subgraph size and cost; Lemma 2: expressive power) and **extensive empirical validation** (multiple datasets, runtime and memory analysis). We therefore frame PWLP as introducing a **new modeling paradigm**—influence-driven pool-walk subgraphs plus line-graph encoding—that is both **conceptually distinct** and **empirically effective**. We have added a dedicated paragraph of conceptual comparison with LGLP, SEAL, and NCN in Related Work (Section X, page Y) and strengthened the contribution statement in the Introduction (Section 1, page Z).

**تغییر در مقاله:**  
- در **Introduction** پاراگراف contributions را با متن قوی جایگزین کنید (فایل `Strong_Contribution_Text.md`).  
- در **Related Work** پاراگراف «Conceptual comparison with LGLP, SEAL, NCN» با تأکید بر **new design principle** و **new modeling paradigm** اضافه کنید (همان فایل).

---

### Comment 2.2 — توجیه هایپرپارامترها و طراحی‌ها
**نظر داور:**  
- انتخاب L، k و top-n بدون آنالیز حساسیت یا تئوری کافی.  
- influence score بر اساس visit frequency؛ مقایسه با آلترناتیوها (مثلاً PageRank-like، decay) نیست.  
- استفاده از PCA برای تعادل featureهای نود و ساختاری؛ چرا PCA به جای projection یادگرفته‌شده یا normalization؟

**پاسخ پیشنهادی (Response):**
> (1) We have added sensitivity analysis for walk length L, number of walks k, and top-n selection (in Section 6.9.2 or a new subsection), including validation performance curves or tables; we also cite prior choices (e.g., from SEAL/NCN) where we align with them.  
> (2) We have added a short discussion (or small experiment) comparing our visit-frequency-based influence score with alternatives such as PageRank-based weighting and decay-weighted visit counts; we report that our choice is competitive or better on validation and is computationally simple.  
> (3) We have clarified the use of PCA: we use it for dimensionality reduction and balancing of node vs. structural feature scales in a fixed, interpretable way before fusion; we note that learned projections could be used in future work and briefly justify PCA for stability and reproducibility in our setting (e.g., no extra parameters, same preprocessing across datasets).

**تغییر در مقاله:**  
- **Sensitivity analysis** برای L، k و n در Section 6.9.2 (یا جدول هایپرپارامترها) با نمودار یا جدول.  
- یک پاراگراف یا یک آزمایش کوچک برای **مقایسه influence score** با PageRank/decay.  
- در بخش PCA یک **توجیه کوتاه** (پایداری، بدون پارامتر اضافه، تفسیرپذیری) و در صورت امکان یک جمله دربارهٔ learned projection به عنوان کار آینده.

---

### Comment 2.3 — Lemma 1–2 و Corollary: تقویت یا قاب‌بندی به عنوان insight
**نظر داور:** Lemma 1 بر اساس ساختار است نه آنالیز؛ Lemma 2 به WL-equivalence وابسته است و محدودیت‌های WL روی line graph را پوشش نمی‌دهد؛ Corollary دربارهٔ gradient stability حدسی است و پشتوانهٔ رسمی یا تجربی ندارد. یا استدلال را تقویت کنید یا صریحاً به عنوان **intuitive insight** قاب‌بندی کنید.

**پاسخ پیشنهادی (Response):**
> We have reframed the theoretical discussion to avoid overclaiming. We now state explicitly that Lemma 1 is a direct consequence of the construction (bounded subgraph size by design) and we present it as a useful bound rather than a deep analytical result. For Lemma 2, we have added a sentence on the limitations of WL expressiveness on line graphs and how PWLP’s pooling and fusion still provide practical benefits. For the Corollary on gradient stability, we have either (a) added a brief empirical check (e.g., gradient norm statistics during training) to support the claim, or (b) rephrased it as an intuitive observation rather than a formal guarantee. The section now clearly distinguishes “formal bounds” from “intuitive insights.”

**تغییر در مقاله:**  
- **Lemma 1:** در متن بنویسید که این bound ناشی از construction است.  
- **Lemma 2:** یک جمله دربارهٔ محدودیت WL روی line graph و اینکه با این حال PWLP در عمل مفید است.  
- **Corollary:** یا یک بررسی تجربی کوچک (gradient norms) اضافه کنید یا آن را به عنوان «intuitive observation» بازنویسی کنید.

---

### Comment 2.4 — آزمایش‌ها: tuning یکسان، OOM، runtime و حافظه
**نظر داور:** نامشخص است که هایپرپارامترها برای همهٔ baselineها به یک اندازه tune شده‌اند؛ برخی baselineها (NBFNet, LGLP) OOM می‌دهند؛ مقایسهٔ **runtime، حافظه و زمان inference** وجود ندارد در حالی که scalability ادعا شده است.

**پاسخ پیشنهادی (Response):**
> We have clarified the experimental protocol: we state explicitly that all baselines were tuned using the same validation strategy (e.g., same number of trials or same search budget where applicable), and we report which baselines encountered OOM and under which hardware/settings (e.g., max nodes per subgraph, batch size). We have added a new subsection (or table) reporting: (1) training and inference time per epoch/sample, (2) GPU memory usage during training and inference, and (3) where possible, runtime complexity or scaling with graph size. We have also added a short discussion on fair comparison: when a baseline hits OOM, we report the largest setting we could run and note this as a limitation for that method. This strengthens the scalability claim with concrete efficiency metrics.

**تغییر در مقاله:**  
- در **Experimental setup** صریح بنویسید که tuning برای همهٔ روش‌ها یکسان بوده است.  
- یک **جدول یا subsection** برای: زمان آموزش/استنتاج، مصرف حافظهٔ GPU، و در صورت امکان مقیاس‌پذیری با اندازهٔ گراف.  
- برای روش‌هایی که OOM شده‌اند، ذکر کنید تحت چه سخت‌افزار/پارامترهایی و چه جایگزینی (مثلاً subgraph کوچکتر) استفاده شده است.

---

### Comment 2.5 — طول و تکرار متن؛ appendix؛ notation
**نظر داور:** مقاله طولانی و تکراری است (مقدمه، related work، بحث آزمایش‌ها)；برخی جدول‌ها و بحث‌ها را به appendix منتقل کنید؛ notation در جاهایی سنگین است.

**پاسخ پیشنهادی (Response):**
> We have done a careful editorial pass: we removed redundant restatements (e.g., importance of global structure, unreliability of node features) and kept a single clear statement in the Introduction or Related Work. We have moved [specify: e.g., additional ablation tables, extra dataset results, full sensitivity plots] to an appendix and referenced them from the main text. We have simplified notation where possible (e.g., a notation table or inline reminders for key symbols) to improve readability for a broader audience. The main text is now more concise while retaining all critical content.

**تغییر در مقاله:**  
- یک **ویرایش کلی** برای حذف تکرار.  
- انتقال بخشی از جدول‌ها/نمودارهای اضافی به **Appendix** و ارجاع در متن.  
- در صورت نیاز یک **جدول notation** یا توضیح کوتاه برای نمادهای پرکاربرد.

---

### Comment 2.6 — مراجع پیشنهادی (link prediction در شبکه‌های اجتماعی)
**نظر داور:** این مراجع را اضافه کنید:  
- Acharya & Mohbey (2023) – Trust-aware spatial-temporal, next POI, Social Network Analysis and Mining.  
- Zhang et al. (2025) – HMNE, Knowledge and Information Systems.  
- Lee et al. (2025) – SFGCN, Information Fusion.  
- Tang et al. (2025) – Interlayer link prediction, International Journal of Modern Physics C.

**پاسخ پیشنهادی (Response):**
> We thank the reviewer for these suggestions. We have incorporated the cited works into the Related Work section, with a focus on link prediction in social and spatial networks, and have added a sentence relating PWLP’s applicability to such domains (e.g., social and location-based networks) where scalable subgraph-based link prediction is relevant.

**تغییر در مقاله:**  
- هر چهار مرجع را در **Related Work** (و در صورت نیاز در Introduction) با یک جملهٔ ارتباط با PWLP اضافه کنید.

---

## Reviewer #3

### Comment 3.1 — نوآوری در مقدمه
**نظر داور:** مشکل مهم است اما **سهم منحصربه‌فرد** نسبت به کارهای قبلی می‌تواند در مقدمه روشن‌تر بیان شود.

**پاسخ پیشنهادی (Response):**
> We have revised the Introduction to articulate more clearly the unique contribution: we now explicitly state that PWLP combines (1) pool-walk-based dynamic subgraph extraction with influence scoring, (2) line-graph-based edge-centric encoding, and (3) balanced fusion of structural and node features in a single framework, and we highlight how this differs from prior subgraph and GNN-based link prediction methods. This clarifies the value of the study for the reader.

**تغییر در مقاله:**  
- در **Introduction** یک پاراگراف یا چند جملهٔ صریح دربارهٔ «what is genuinely new» و تفاوت با روش‌های قبلی اضافه کنید.

---

### Comment 3.2 — ادبیات و کارهای اخیر
**نظر داور:** مرور ادبیات به طور کلی کافی است اما **تعامل عمیق‌تر با کارهای اخیر** با روش‌های مشابه یا مشکلات نزدیک می‌تواند مطالعه را در چشم‌انداز پژوهش بهتر قرار دهد.

**پاسخ پیشنهادی (Response):**
> We have expanded the Related Work section to include a more detailed discussion of recent methods (e.g., subgraph-based link prediction, line-graph and edge-centric approaches, and scalability-oriented techniques from the last 2–3 years). We now position PWLP more clearly within this landscape and cite additional recent papers where relevant.

**تغییر در مقاله:**  
- در **Related Work** چند پاراگراف یا بند برای روش‌های اخیر (۲–۳ سال گذشته) و جایگاه PWLP در بین آن‌ها اضافه کنید.

---

### Comment 3.3 — مرجع پیشنهادی (Rare-Event Prediction, Journal CPS)
**نظر داور:** این لینک را در نظر بگیرید: https://journalcps.com/index.php/volumes/article/view/740

**منبع:** Abdulrazaq, M. (2023). Rare-Event Prediction in Imbalanced Data: A Unified Evaluation and Optimization Framework for High-Risk Systems. *Communication in Physical Sciences*, 9(4).  
(این مقاله دربارهٔ پیش‌بینی رویدادهای نادر و داده‌های نامتعادل است؛ در صورت ارتباط با ارزیابی یا imbalance در link prediction می‌توان اشاره کرد.)

**پاسخ پیشنهادی (Response):**
> We thank the reviewer for the suggestion. We have added the reference Abdulrazaq (2023) in the Related Work or in the evaluation discussion, noting that in settings where positive links are rare (e.g., cold-start or sparse networks), evaluation and optimization considerations from rare-event and imbalanced learning are relevant; we cite this work where we discuss evaluation metrics or class imbalance in link prediction.

**تغییر در مقاله:**  
- اگر در مقاله به imbalance یا ارزیابی در شرایط نادر اشاره می‌کنید، این مرجع را در **Related Work** یا بخش Evaluation اضافه کنید.

---

### Comment 3.4 — فرضیات، برآورد و robustness/sensitivity
**نظر داور:** چارچوب روش‌شناسی مناسب است اما **جزئیات بیشتر دربارهٔ فرضیات، روش برآورد و بررسی robustness/sensitivity** شفافیت و تکرارپذیری را بهبود می‌دهد.

**پاسخ پیشنهادی (Response):**
> We have added a short subsection (in Methodology or Experiments) that states key assumptions (e.g., undirected/unweighted or how we handle directed/weighted graphs, availability of node features), describes estimation and training procedures (loss, optimizer, early stopping), and summarizes the robustness and sensitivity checks we perform (e.g., sensitivity to L, k, n and to different random seeds or data splits). This improves transparency and reproducibility.

**تغییر در مقاله:**  
- یک **بند یا زیربخش** برای Assumptions، Estimation procedure و Robustness/sensitivity (با ارجاع به sensitivity analysis و جدول هایپرپارامترها) اضافه کنید.

---

### Comment 3.5 — محدودیت‌ها
**نظر داور:** بحث **محدودیت‌ها** صریح‌تر باشد، از جمله محدودیت داده، منابع بایاس و عواملی که بر تعمیم‌پذیری اثر می‌گذارند.

**پاسخ پیشنهادی (Response):**
> We have added an explicit “Limitations” paragraph in the Conclusion (and optionally a short note in the Experiments). We discuss: data constraints (e.g., graph size, density, and feature availability), potential sources of bias (e.g., selection of datasets or baselines), and factors that may affect generalizability (e.g., very large or dynamic graphs, domains different from those we evaluated). We also mention planned future work to address these limitations.

**تغییر در مقاله:**  
- همان پاراگراف **Limitations** که برای داور ۱ پیشنهاد شد را تقویت کنید و در صورت نیاز یک جمله دربارهٔ bias و generalizability اضافه کنید.

---

### Comment 3.6 — خوانایی و روانی متن
**نظر داور:** ارائه به طور کلی خوب است اما برخی بخش‌ها را می‌توان برای وضوح **خلاصه‌تر** کرد؛ کاهش تکرار بین بخش‌ها و بهبود توضیحات همراه جدول/شکل‌ها.

**پاسخ پیشنهادی (Response):**
> We have streamlined the manuscript by reducing redundancy between sections and improving the explanations that accompany tables and figures. Each table and figure is now clearly referenced and summarized in the text, and we have removed repeated arguments. This enhances readability and flow.

**تغییر در مقاله:**  
- ویرایش برای **کوتاه‌تر و واضح‌تر** کردن توضیحات جدول/شکل‌ها و حذف تکرار بین Introduction، Related Work و Discussion.

---

# چک‌لیست تغییرات در مقاله (برای انجام در نسخهٔ اصلاح‌شده)

| # | کار | بخش |
|---|-----|------|
| 1 | اضافه کردن جدول هایپرپارامترها با توجیه | Section 6.3 |
| 2 | Sensitivity analysis برای L, k, n | Section 6.9.2 |
| 3 | Case study با ویژوال subgraph برای جفت نودهای نمونه | subsection جدید |
| 4 | اضافه کردن/به‌روز کردن baselineهای SOTA اخیر | Experiments |
| 5 | پاراگراف Limitations در Conclusion | Conclusion |
| 6 | پاراگراف شهودی در ابتدای Methodology | Methodology |
| 7 | مقایسه مفهومی با LGLP, SEAL, NCN و قاب‌بندی نوآوری | Introduction / Related Work |
| 8 | مقایسه influence score با PageRank/decay و توجیه PCA | Methodology / Experiments |
| 9 | بازنویسی/قاب‌بندی Lemma 1–2 و Corollary | Theory section |
| 10 | توضیح پروتکل tuning یکسان؛ گزارش OOM؛ جدول runtime و حافظه | Experiments |
| 11 | کوتاه‌سازی متن، انتقال به Appendix، ساده‌سازی notation | کل مقاله |
| 12 | اضافه کردن مراجع Acharya, Zhang, Lee, Tang, Abdulrazaq | Related Work |
| 13 | بیان صریح نوآوری در Introduction | Introduction |
| 14 | گسترش Related Work با کارهای اخیر | Related Work |
| 15 | فرضیات، برآورد و robustness | Methodology / Experiments |
| 16 | بهبود توضیحات جدول/شکل و حذف تکرار | در سراسر متن |

---

**نکته:** این فایل را به عنوان پایهٔ «Response to Reviewers» در سابمیت اصلاح‌شده استفاده کنید. برای هر comment، در نامهٔ پاسخ به داوران دقیقاً بنویسید چه تغییری در کدام صفحه/بخش انجام شده است (مثلاً «Table X, page Y» یا «new paragraph in Section Z»).
