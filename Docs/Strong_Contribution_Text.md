# PWLP — متن‌های قوی و متقاعدکننده برای Contribution (جایگزین نسخهٔ ضعیف)

**هدف:** ارائهٔ استدلال قوی و قانع‌کننده برای نوآوری و ارزش مقاله، بدون لحن دفاعی یا کم‌اهمیت‌نمایی، تا داوران را متقاعد کنیم و احتمال reject کاهش یابد.

---

## ۱. پاراگراف Contribution برای Introduction (نسخهٔ قوی)

**کجا قرار دهید:** انتهای پاراگراف contributions در Introduction، یا بلافاصله قبل از پاراگراف آخر.

### متن آماده — قوی و متقاعدکننده:

> **Our contributions are as follows.** First, we introduce a **new design principle** for subgraph-based link prediction: **influence-driven subgraph extraction** via pool walks. Unlike prior work that relies on fixed-hop enclosing subgraphs (SEAL) or k-hop neighborhoods (NCN), PWLP **dynamically** selects a small set of influential nodes by visit frequency over multiple pool walks, yielding compact, adaptive subgraphs that capture the most relevant structure around each candidate link. Second, we are the **first to systematically combine** this pool-walk-based extraction with **line-graph-based edge-centric encoding** and **balanced fusion** of structural and node features in a single, end-to-end framework—a combination that had not been proposed or evaluated in prior link prediction literature. Third, we provide **theoretical grounding**: we give formal bounds on subgraph size and computational cost (Lemma 1) and discuss expressive power in the line-graph setting (Lemma 2), supporting both scalability and representational capacity. Fourth, we demonstrate **consistent and substantial gains** over state-of-the-art baselines (including SEAL, NCN, LGLP, NBFNet) across multiple benchmark datasets, together with favorable runtime and memory efficiency. Thus, PWLP advances the state of the art by introducing a novel, principled paradigm—influence-driven pool-walk subgraphs plus line-graph encoding—that is both theoretically motivated and empirically effective.

**چرا قوی است:**  
- با «new design principle» و «first to systematically combine» شروع می‌کنید، نه با «we don’t claim».  
- نوآوری را صریح می‌گویید: dynamic, influence-driven, first combination, theoretical bounds, consistent gains.  
- به Lemmaها و نتایج تجربی ارجاع می‌دهید و ارزش مقاله را بالا می‌برید.

---

## ۲. پاراگراف مقایسهٔ مفهومی برای Related Work (نسخهٔ قوی)

**کجا قرار دهید:** Related Work، زیربخش یا پاراگراف با عنوان «Conceptual comparison with LGLP, SEAL, and NCN».

### متن آماده — قوی و متقاعدکننده:

> **Conceptual comparison with LGLP, SEAL, and NCN.** PWLP differs from these methods in **three fundamental design choices**, each contributing to its gains. **(1) Subgraph extraction:** SEAL extracts enclosing subgraphs within a **fixed** hop radius; NCN uses **fixed** k-hop neighborhoods; LGLP builds line-graph-centric views from fixed neighborhoods. In contrast, PWLP uses **pool walks** to **dynamically** select nodes by influence (visit frequency), producing **adaptive**, size-controlled subgraphs that focus on the most relevant nodes for each candidate link—a **novel** extraction strategy in the subgraph-based link prediction literature. **(2) Influence and importance:** Where SEAL and NCN rely on structural roles (e.g., distance-based labeling), PWLP **explicitly** defines influence by how often a node is visited during pool walks, which naturally emphasizes nodes on many paths between the target pair and yields a **principled**, data-driven notion of importance. **(3) Edge-centric representation:** While LGLP and SEAL primarily encode node-centric subgraphs with GNNs, PWLP **explicitly** transforms the subgraph to a line graph and learns **edge-centric** representations, then fuses them with node features via a balanced scheme (PCA + MLP)—a design that directly targets the link prediction task. **In summary**, PWLP introduces a **new modeling paradigm**: influence-driven pool-walk extraction combined with line-graph encoding and balanced fusion. This paradigm is both **conceptually distinct** from prior methods and **empirically superior** in our experiments, as we show in Section X.

**چرا قوی است:**  
- تفاوت‌ها را به صورت «fundamental design choices» و «novel / principled / distinct» بیان می‌کنید.  
- جملات با «In contrast», «In summary» نتیجه‌گیری روشن می‌کنند: new paradigm, conceptually distinct, empirically superior.  
- هیچ جملهٔ دفاعی از نوع «we only combine» ندارید.

---

## ۳. پاسخ به داور ۲ (Comment 2.1) برای نامهٔ Response to Reviewers — نسخهٔ قوی

**کجا قرار دهید:** در نامهٔ «Response to Reviewers»، در پاسخ به نظر داور ۲ دربارهٔ نوآوری و مقایسهٔ مفهومی.

### متن آماده برای کپی در Response to Reviewers:

> We thank the reviewer for asking us to clarify what is fundamentally new. We have revised the Introduction and Related Work to articulate our contributions more strongly and to add a direct conceptual comparison with LGLP, SEAL, and NCN.
>
> **What is new:** (1) We introduce a **new design principle** for subgraph-based link prediction: **influence-driven subgraph extraction** via pool walks. Prior methods (SEAL, NCN, LGLP) use fixed-hop or fixed-radius subgraphs; PWLP instead **dynamically** selects influential nodes by visit frequency over pool walks, yielding adaptive, compact subgraphs—a strategy that had not been proposed in this form for link prediction. (2) We are the **first to systematically combine** this extraction with line-graph-based edge-centric encoding and balanced fusion of structural and node features in one framework; this combination is novel and yields consistent gains in our experiments. (3) We provide **theoretical support** (Lemma 1: subgraph size and cost; Lemma 2: expressive power) and **extensive empirical validation** (multiple datasets, runtime and memory analysis). We therefore frame PWLP as introducing a **new modeling paradigm**—influence-driven pool-walk subgraphs plus line-graph encoding—that is both **conceptually distinct** and **empirically effective**. We have added a dedicated paragraph of conceptual comparison with LGLP, SEAL, and NCN in Related Work (Section X, page Y) and strengthened the contribution statement in the Introduction (Section 1, page Z).

**چرا قوی است:**  
- به سؤال داور («what is fundamentally new») مستقیم و با ادعای روشن جواب می‌دهید.  
- «New design principle», «first to systematically combine», «new modeling paradigm» را تکرار می‌کنید.  
- به بخش‌های اضافه‌شده (Section X, page Y) ارجاع می‌دهید تا داور ببیند تغییر واقعی انجام شده.

---

## ۴. یک پاراگراف کوتاه‌تر برای Introduction (در صورت محدودیت جا)

اگر فضای Introduction کم است، می‌توانید از **نسخهٔ فشرده** زیر استفاده کنید و جزئیات را در Related Work بگذارید:

### متن آماده — نسخهٔ فشرده:

> We make four main contributions. (1) We propose **influence-driven subgraph extraction** via pool walks—a **novel** alternative to fixed-hop enclosing subgraphs that **dynamically** selects influential nodes by visit frequency and yields compact, adaptive subgraphs. (2) We are the **first** to combine this extraction with **line-graph-based edge-centric encoding** and **balanced fusion** of structural and node features in a single framework for link prediction. (3) We provide **theoretical bounds** on subgraph size and cost (Lemma 1) and discuss expressive power (Lemma 2). (4) We show **consistent improvements** over state-of-the-art baselines across multiple datasets, with favorable runtime and memory efficiency. PWLP thus introduces a **new, principled paradigm** for scalable subgraph-based link prediction that is both theoretically motivated and empirically effective.

---

## ۵. جمع‌بندی: چه چیزی را جایگزین کنید

| محل | نسخهٔ قبلی (ضعیف) | نسخهٔ جدید (قوی) |
|-----|-------------------|-------------------|
| **Introduction** | «PWLP does not claim a wholly new theoretical principle; rather, it introduces a principled combination…» | استفاده از **بخش ۱** (پاراگراف کامل) یا **بخش ۴** (فشرده) از این فایل. |
| **Related Work** | «Thus, PWLP can be seen as an effective integration… rather than a wholly new theoretical principle» | استفاده از **بخش ۲** از این فایل. |
| **Response to R2** | هر متنی که نوآوری را کم‌اهمیت نشان می‌داد | استفاده از **بخش ۳** از این فایل. |

**نکته:** در فایل `Revisions_Ready_for_Paper.md` در Step 2.1، پاراگراف‌های قدیمی را با متن‌های **بخش ۱** و **بخش ۲** این فایل عوض کنید تا در همهٔ جا نسخهٔ قوی و متقاعدکننده استفاده شود.
