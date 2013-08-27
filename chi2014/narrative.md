TODO
====

+ Sit down with Mike's graphs from Pilot 11 - 13 to pick out anything of interest, and also to see which questions may be in part answered.
+ Pilot 11 - 13: Some questions seem to have a stronger proportion of original ideas than others. This can be tested further (need to plot multiple questions on same y scale).
+ see if "footprints" are worth keeping as data
+ compute inmix/manual originality correlation on per-question basis
	+ also, compute correlation of the actual measures and not just the summary stats


rhetoric
========

## Our own start

We wanted to *automatically* solve a problem that required *creativity*, which a computer can't provide. In creating a workflow that attempted to use a crowd marketplace as a creativity source, we made many *assumptions*. We realized we were doing this, and found other work that did it as well. We tried to find *guidelines* to inform our assumptions, and realized they didn't exist.

## Design implications

Make better brainstorming tasks to generate creativity.



assumptions
===========

1. **brainstorming** is a representative creativity-generation task
2. **nominal brainstorming** is not only sufficient, but in fact empirically superior to real brainstorming. Thus, it is a sufficient basis for a crowd brainstorming system.
3. When designing brainstorming tasks, we fix certain parameters of the design space (relevance, rules) to those recommended in brainstorming guidelines



questions
=========

**Bold** if examined by us in some way

1. **What is the design space of a brainstorming task?**
2. **What is the output space of a brainstorming task?**
3. How does manipulating a parameterization of (1) affect an output's position in (2)?
	+ **problem -> originality**
	+ **problem -> completion time**
	+ **problem -> remix**
	+ problem -> utility
	+ problem -> creativity
	+ **problem -> run: number of category leaves**
	+ problem -> run: number of category parents
	+ **number of responses solicited -> originality**
	+ **number of responses solicited -> completion time**
	+ **number of responses solicited -> remixing**
	+ number of responses solicited -> utility
	+ number of responses solicited -> realisticness
	+ number of responses solicited -> creativity
	+ number of responses solicited -> run: number of category leaves
	+ number of responses solicited -> run: number of category parents
	+ **number of runs -> run: number of category leaves**
	+ number of runs -> run: number of category parents
4. Within (1), what separates a core axis from an *intervention*? What interventions could we introduce to brainstorming tasks?
5. What, if any, structure of (2) is independent of (1)'s parameterization?
6. Are any axes in 2 dependent? 
	+ time spent on response <-> originality
	+ time spent on response <-> utility
	+ time spent on response <-> creativity
	+ time spent on response <-> introduction of new category leaf
	+ time spent on response <-> introduction of new category parent
	+ **time spent on response <-> order**
	+ word count <-> originality
	+ word count <-> utility
	+ word count <-> creativity
	+ **word count <-> order**
	+ **order <-> originality**
	+ order <-> utility
	+ order <-> creativity
	+ order <-> remix
	+ **originality <-> remix**
	+ originality <-> realisticness
	+ originality <-> utility
	+ remix <-> introduction of new category leaf
	+ remix <-> introduction of new category parent
7. Fixation
	+ Do people use a single strategy across an entire brainstorming session, or do they switch stragies? If the latter, where/when do they switch?
	+ At what level in a category heirarchy of ideas do people tend to have thier root? Does everyone go to the top or do some stay in sub-trees?
8. What is a reasonable measure for "creativity"?
10. Can we model the creativity of a person as a parameter? How long does it take for this model to converge enough to make a judgment?
11. What is a good model for...
	+ rate of new (category heirarchy leaves) ideas
	+ rate of new design dimensions
	+ rate of new (category heirarchy parents of leaves) categories
	+ time spent on each response
12. What are good measures of originality?
	+ o-score?
	+ manual coding?
	+ inverse cluster size?
	+ inverse sum similarity?
13. What is a good methodology for automatically categorizing ideas?
	+ what is a good similarity metric for ideas?
14. **Can we automate measurement of originality?**
15. What kind of "personas" do we see in the data?


history
=======

Crowd brainstorming was a problem we needed to make judgments about, and we discovered there were no guidelines to do so. We set out to discover those guidelines.

## Pilot1
+ Initially, we piloted with a brainstorming question that we already had: "List things that could cause a change in body weight".
	+ we solicited 5, 10, and 20 size response sets
	+ found a 1/x-ish graph of decay in the number of times ideas were proposed
	+ hypothesized that as you sampled more and more runs, the total number of unique ideas you had would be roughly logarithmic
		+ in actuality, lines seemed to fit the data better than logarithms.
	+ hypothesized: larger sets would see more common answers
		+ tested this by counting the mean number of "top 5" most-common ideas globally in a set
		+ in actuality, this was roughly uniform across size conditions (2)
	+ tested: how is the "o-score" (1-p(idea)) related to the ordinal position of the response? (ref Krzysztof, Pao)
		+ in general, o-score increases as ordinal position increases (looks roughly logarithmic). Most visible in 20 size set.

## Pilot5
+ After this, we upped our questions to three (mop, weight, thumb) on the basis of using more canonical brainstorming questions. We also added a size-50 condition.
	+ Fil looked at word counts and time spent across question conditions, size conditions, and order
		+ question conditions seemed to have an affect on word count, time spent
		+ size conditions seemed to have an affect on time spent (Mike said linear)
		+ order seemed to have no effect on either
	+ Mike did a significant amount of descriptive stats on this data, from which only his email summaries have been taken right now
+ A subset of the data was coded for creativity, realisticness, surprise factor, thematic similarity to previous answer - this exposed two things: our unhappiness with the naive coding scheme, and our unhappiness with the quality of answer to our questions
+ As researchers, we brainstormed questions we thought would have higher relevance to turk users. We settled on a subset of 5, and through self-tests at brainstorming settled on four: turk, charity, mp3 and forgot_name

## Pilots 6 - 10
Paid more and changed the questions. Tweaked question wording and coding scheme until we were happy with them. 

+ coding scheme reduced to creativity and underdefined, and eventually expanded to originality (3 point), realisticness (3 point), utility (3 point), distance from similar(Z+)
+ finalized question scheme also included a text box where brainstormers could add any additional ideas

## Pilots 11 - 13
These are all using the final questions, across a test run, large run, and bug fix run. After the final run we finally discarded the "as many as possible" condition when it became obvious we were getting not enough responses and that they were of poor quality, even when changing the reward amount.

New terminology: inmix/outmix

+ Fil and Mike coded responses from these conditions on the finalized coding scheme
+ in this phase, there was more graphical analysis of the automatically detectable fields (word count, time spent)
+ from here, progress began on NLP tools to automate parts of the coding process:
	+ Similarity scores
		+ levenshtein
		+ tf-idf (document is corpus of all answers in this case, term is the idea)
	+ clustering (categorization)
		+ correlation clustering (algorithm cautious)
+ from this, we have three originality scores:
	+ inverse cluster size
	+ inverse sum similarity
	+ manual (codes > 0 for both coders)
	+ manual, no riffing (codes > 0 for both, not an outmix)
+ clusters were touched up manually
	+ this process was difficult and error prone, and we are developing a rool to speed it up
+ attempted Good Turing frequency estimation on the idea pool to try and predict probability of seeing a new idea. First pass at this failed spectacularly

## Pilot 14

+ thought there was a linear relationship between originality and number of responses requested; convinced this had to fall off somewhere
+ requested old questions with 75/100 responses
+ coded this data and previous in heirarchical clusters

our answers
===========

## 1 What is the design space of a brainstorming task?

Question:

+ problem
+ context/motivation
+ constraints

Task:

+ question
+ number of responses solicited
+ training
+ instruction
+ reward

Soliciation:

+ number of runs

## 2 What is the output space of a brainstorming task?

These are composed of measures we can take, and ideal measures we'd like to take.

Response-scale:

+ word count
+ completion time
+ originality
+ "remix" degree
+ utility
+ realisticness
+ creativity
+ category heirarchy
+ design dimensions

Run-scale:

+ summary metrics for any response-scale metrics
+ ordering of responses

## 3. How does manipulating a parameterization of (1) affect an output's position in (2)?

Increasing the *number of responses requested* has an increasing effect on originality, even normalized. Allowing people to choose their own number of responses results in poor responses regardless of reward (for fixed rewards, we did not try scaling).

### problem -> originality
By both manual and ISS orignality scores, the turk and forgot_name questions generate significantly less original responses than the other two questions. This does not show up in the h-cluster o-score measure (probably because these o-scores are normalized to probability of response given question, not overall probability

Should we be computing probability based on all responses to all questions, or on a per-question basis?


### number of runs -> run: number of category leaves

+ pilot1: hypothesized that as you sampled more and more runs, the total number of unique ideas you had would be roughly logarithmic
	+ in actuality, lines seemed to fit the data better than logarithms.

### problem -> completion time
+ seemed to find an affect in pilot5

### number of responses solicited -> completion time
Pilot6: mike found a roughly linear relationship

### problem -> run: number of category leaves
Pilot 11 - 13: No real connection visible here. Category leaves are clusters, and we see about the same proportion across cluster sizes between questions.

### problem -> originality
Pilot 11 - 13: Some questions seem to have a stronger proportion of original ideas than others. This can be tested further (need to plot multiple questions on same y scale).

### number of responses solicited -> originality
Pilot 11 - 13: Non-linear growth relationship here, but it only holds for *certain questions*. As the number of responses solicited grows, it also seems that the distribution of *number of original responses received* flattens out.

### number of responses solicited -> remixing
Pilot 11 - 13: At least for the 50 case, number of inmixes goes way up (significantly) (across all questions). As the number of responses solicited grows, it also seems that the distribution of *inmixes* flattens out.

### problem -> remix
Pilot 11 - 13: iPod and forget_name questions show significant differences in inmix that increase across number requested condition. Other questions do not. This suggests questions have some effect.

inmix: outmix ratios looks very different across problem conditions, which suggests that problem has an effect. However, this seems to normalize when looking only at the 50 condition.

### **Can we automate measurement of originality?** ###

Both automated measures of originality correlate poorly with manual measures. The inverse cluster size does show singificant increase in originality as a function of number of responses solicited, which at least has the right ordinal shape as the same with manual measures.

## 4. Within (1), what separates a core axis from an *intervention*? What interventions could we introduce to brainstorming tasks?

Interventions:

+ facilitation
+ training
+ How does previous exposure to brainstorming tasks (same problem, different problem) affect future performance?
+ ask people to identify categories of ideas
+ prevent submission until X time has passed since last idea

## 6. Are any axes in 2 dependent?

### order <-> originality
+ tested: how is the "o-score" (1-p(idea)) related to the ordinal position of the response?
	+ in general, o-score increases as ordinal position increases (looks roughly logarithmic). Most visible in 20 size set.

### word count <-> order
Pilot5 tested this and found nothing.

### time spent on response <-> order
Pilot5 tested this and found nothing.

### originality <-> remix
Found a 0.68 - 0.78 (depending on whether riffs are counted as original) correlation between originality and inmix summary statistics for runs (i.e. # of original ideas and # of inmix ideas).




other's answers
===============

## 1 What is the design space of a brainstorming task?

Question:

+ difference "sizes" of brainstorming problems (Zagona 1966):
	* explanation problems
	* prediction problems
	* invention problems
+ "remoteness" (Zagona)



## 3 How does manipulating a parameterization of (1) affect an output's position in (2)?

linear growth between number of runs and number of ideas (Bouchard 1970)

## 5  What, if any, structure of (2) is independent of (1)'s parameterization?

"more good ideas were found in the final third of a subject's list of ideas than in the first two thirds of the list" (Zagona 1966)


## 8
Yu/Nickerson "Finke's method": originality x f(utility, realisticness)
