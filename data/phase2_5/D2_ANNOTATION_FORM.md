# D2 annotation — 20 claim pairs

**Annotator:** `__________________`  ← put your name here before you start

**Do not discuss these with the other annotator.** You are being compared, and
the comparison is the point.

---

## What you are deciding

Two claims from the same conversation. Pick **exactly one** of four:

| Code | Meaning |
|---|---|
| `I` | **INDEPENDENT** — different facts, no relation |
| `U` | **DUPLICATE** — the same speaker restating the same fact |
| `C` | **CONFLICT** — incompatible claims about the same fact, no update |
| `S` | **SUPERSEDES** — the same fact updated; the later claim replaces the earlier |

## Four rules

1. **An assistant echoing the user is `I`.** A user decision plus an assistant
   endorsement are two assertions by two speakers. Agreeing does not duplicate
   or retire anything.
2. **`S` needs the same fact updated.** "I weigh 70kg" → "68kg" is `S`. Two
   different objects are `I` however similar the advice sounds.
3. **`C` needs incompatible claims**, not a shared topic.
4. **Sharing a subject is not sharing a fact.** Two properties of one lens are
   two facts. A question and its answer are two things, not one.

**When in doubt, `I` is the correct default.** `U`, `C` and `S` each make a
strong claim; if you cannot name the single shared proposition, none applies.

## How to answer

Replace the `_` after **ANSWER** with `I`, `U`, `C` or `S`. Add a note after
**NOTE** whenever you hesitate — the notes are as useful as the labels.

---

### 1. `d2f_0000`

*knowledge-update · cross-session*

**A** — [user] 2023-07-16 07:07
> I'm considering applying for a green card, but I'm not sure about the process and requirements.

**B** — [user] 2023-10-20 00:00
> My parents have been a huge help in preparing all the paperwork, they've been staying with me for nine months now.

**ANSWER:** `_`

**NOTE:**

---

### 2. `d2f_0001`

*multi-session · cross-session*

**A** — [user] 2023-05-20 20:05
> it's been losing some leaves, but I've read that's normal

**B** — [user] 2023-05-24 12:30
> My peace lily has been losing leaves since I brought it home.

**ANSWER:** `_`

**NOTE:**

---

### 3. `d2f_0002`

*single-session-user · same session*

**A** — [user] 2023-05-26 10:26
> I was thinking of trying out some BBQ ribs for my party, and I remember Alex telling me he marinated them in a special sauce for 24 hours before grilling them to perfection.

**B** — [assistant] 2023-05-26 10:26
> BBQ ribs are a crowd-pleaser, and marinating them in a special sauce for 24 hours will definitely make them tender and flavorful.

**ANSWER:** `_`

**NOTE:**

---

### 4. `d2f_0003`

*multi-session · cross-session*

**A** — [assistant] 2023-05-20 20:05
> Peace lilies prefer a humid environment, but they can adapt to average humidity levels.

**B** — [user] 2023-05-24 12:30
> My peace lily has been losing leaves since I brought it home.

**ANSWER:** `_`

**NOTE:**

---

### 5. `d2f_0004`

*knowledge-update · same session*

**A** — [user] 2023-05-30 22:16
> I need help organizing my shoe closet because it's getting hard to find what I need.

**B** — [assistant] 2023-05-30 22:16
> organizing your shoe closet can help you find what you need and make your life easier.

**ANSWER:** `_`

**NOTE:**

---

### 6. `d2f_0005`

*knowledge-update · same session*

**A** — [user] 2023-03-11 03:12
> I've been having some issues with my old 18-55mm kit lens, and I've been relying on manual focus lately.

**B** — [user] 2023-03-11 03:12
> I'm mostly using my camera for portrait and low-light photography, so I think I'll need a bag that can protect my gear well.

**ANSWER:** `_`

**NOTE:**

---

### 7. `d2f_0006`

*multi-session · cross-session*

**A** — [user] 2023-05-20 20:05
> The user is unsure about how often to mist their peace lily.

**B** — [user] 2023-05-24 12:30
> My peace lily has been losing leaves since I brought it home.

**ANSWER:** `_`

**NOTE:**

---

### 8. `d2f_0007`

*multi-session · same session*

**A** — [assistant] 2023-05-20 20:05
> It's important to understand the type of fertilizer and its concentration to ensure it's suitable for the peace lily.

**B** — [assistant] 2023-05-20 20:05
> Peace lilies prefer a humid environment, but they don't require as much misting as ferns do.

**ANSWER:** `_`

**NOTE:**

---

### 9. `d2f_0008`

*knowledge-update · cross-session*

**A** — [assistant] 2023-06-18 00:59
> I'm happy to help you with staying motivated on lazy Sundays.

**B** — [user] 2023-09-02 01:13
> I've been using my Fitbit Charge 3 for 9 months now.

**ANSWER:** `_`

**NOTE:**

---

### 10. `d2f_0009`

*multi-session · same session*

**A** — [assistant] 2023-05-21 19:38
> Select relevant features that are meaningful for clustering. Remove features with low variance or high correlation with other features.

**B** — [assistant] 2023-05-21 19:38
> Transform categorical features into numerical features using techniques like one-hot encoding or label encoding.

**ANSWER:** `_`

**NOTE:**

---

### 11. `d2f_0010`

*single-session-user · same session*

**A** — [user] 2023-05-26 10:26
> Alex made amazing BBQ ribs at his party a few weeks ago, and they were tender and flavorful.

**B** — [user] 2023-05-26 10:26
> I was thinking of trying out some BBQ ribs for my party, and I remember Alex telling me he marinated them in a special sauce for 24 hours before grilling them to perfection.

**ANSWER:** `_`

**NOTE:**

---

### 12. `d2f_0011`

*knowledge-update · same session*

**A** — [assistant] 2023-04-11 21:07
> Matsumoto is a wonderful base for exploring the Japanese Alps and experiencing traditional Japanese culture.

**B** — [assistant] 2023-04-11 21:07
> Matsumoto is a treasure trove of traditional Japanese architecture and history. Here are some must-visit spots to add to your itinerary:

**ANSWER:** `_`

**NOTE:**

---

### 13. `d2f_0012`

*multi-session · same session*

**A** — [user] 2023-05-20 20:05
> it's been losing some leaves, but I've read that's normal

**B** — [user] 2023-05-20 20:05
> The user is unsure about how often to mist their peace lily.

**ANSWER:** `_`

**NOTE:**

---

### 14. `d2f_0013`

*multi-session · cross-session*

**A** — [user] 2023-05-20 20:05
> The ideal temperature range for a peace lily is between 65°F to 70°F (18°C to 21°C).

**B** — [user] 2023-05-24 12:30
> My peace lily has been losing leaves since I brought it home.

**ANSWER:** `_`

**NOTE:**

---

### 15. `d2f_0014`

*knowledge-update · same session*

**A** — [user] 2023-03-11 03:12
> I recently got a new 50mm prime lens, which has been working out great.

**B** — [user] 2023-03-11 03:12
> I'm mostly using my camera for portrait and low-light photography, so I think I'll need a bag that can protect my gear well.

**ANSWER:** `_`

**NOTE:**

---

### 16. `d2f_0015`

*knowledge-update · cross-session*

**A** — [user] 2023-01-21 15:05
> Maybe they can suggest some books that will help me get back into 'The Nightingale'.

**B** — [user] 2023-03-30 01:14
> I just finished reading 'The Nightingale' by Kristin Hannah, which was amazing, by the way.

**ANSWER:** `_`

**NOTE:**

---

### 17. `d2f_0016`

*multi-session · same session*

**A** — [user] 2023-05-20 20:05
> The user is unsure about how often to mist their peace lily.

**B** — [assistant] 2023-05-20 20:05
> Water your peace lily when the top 1-2 inches of the soil feel dry to the touch. This is usually every 7-10 days during the spring and summer months when the plant is actively growing. During the fall and winter, you can reduce watering to every 10-14 days, as the plant grows slower.

**ANSWER:** `_`

**NOTE:**

---

### 18. `d2f_0017`

*temporal-reasoning · same session*

**A** — [assistant] 2023-05-23 08:02
> The 50mm f/1.8 is a great lens for portraits and mastering it will help improve portrait photography.

**B** — [assistant] 2023-05-23 08:02
> The 50mm lens can exhibit some distortion, especially when shooting at close range.

**ANSWER:** `_`

**NOTE:**

---

### 19. `d2f_0018`

*knowledge-update · cross-session*

**A** — [user] 2023-07-16 07:07
> common mistakes people make during the application process that I should avoid

**B** — [user] 2023-10-20 00:00
> My parents have been a huge help in preparing all the paperwork, they've been staying with me for nine months now.

**ANSWER:** `_`

**NOTE:**

---

### 20. `d2f_0019`

*temporal-reasoning · same session*

**A** — [assistant] 2023-05-23 08:02
> The 50mm f/1.8 is a great lens for portraits and mastering it will help improve portrait photography.

**B** — [assistant] 2023-05-23 08:02
> The 50mm f/1.8 lens is a great choice for portraits and can create a beautiful bokeh effect.

**ANSWER:** `_`

**NOTE:**

---
