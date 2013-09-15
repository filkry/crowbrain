# CHI 2014: Brainstorming

## Introduction

### Motivation

+ our own goals

+ personas and their goals
+ potential goals in brainstorming
	+ as many distinct ideas as possible
	+ many categories of idea
	+ emphasize the originality
	+ emphasize creativity - which may mean a great number of things
	+ get ideas as quickly as possible
	+ specificity or generality

+ design implications

### Establish goals

+ Establish a corpus of brainstorming data for straightforward tasks (extremely restricted design space)
+ within this design space, how do outcomes vary?
	+ saturation
	+ originality
	+ specificity
	+ utility?
	+ surprise?
	+ creativity? (Finke's?)
	+ fixation
	+ time spent on responses
+ establish a method for categorizing ideas?

### Summarize findings?

## Related work

Organize this in the same way we organize the findings section of the paper.

+ general presence of intervention schemes in crowd brainstorming research
	+ Yu/Nickerson 2011

+ Bouchard 1970 - linear growth between the number of people in nominal growths and the number of ideas

+ quantity of ideas is positively correlated with the quality of ideas (Diehl and stroebe)

+ more you torture, better results (on subject rating of uniqueness and value)

+ SIAM model described what happens in a run
	It predicts that the ideas generated in sequence are likely to be activated by the same mental image (in this case same category). 
		+ specific paper hypothesis: An idea should be more likely followed by the same idea than would be expected according to chance
		+ tested with ARC
	New ideas within the same categories come faster than when you change categories.
		+ ARC is correlated highly with production. In our case, ARC would be correlated negatively with total time spent?
		+ also found significant difference between time spent in category changes and time spent out of category change
	It is expected that you can revisit a category after switching to others. However, it doesn't say whether the come back is for elaboration or you find a new idea.
	The change of categories occur (probabilistically) when you exhaust ideas from the current image in your working memory.

+ group brainstorming vs nominal brainstorming (Taylor, Berry Block 1958)
	+ nominal brainstorming performs superior
+ electronic real group brainstorming: Gallupe 1992
	+ reduces the social inhibants of group brainstorming (production blocking, simultaneous responses)
	+ nominal brainstorming with exposure to shared ideas (input into WM in terms of the SIAM model)'
+ groups perform better in electronic environements (Dennis and Valacich)
+ counterpoint: crowd marketplaces make no guarantees of concurrency; to that end nominal is a reasonable default position

+ repeat exposure increases creativity (where did I find this?) (Should go closer to actual comparison?)

+ measures of creativity
	+ we decided to focus on the components of creativity in our analysis

+ Methods for measuring ideation effectiveness
	+ breaks down into novelty, variety (within a set of ideas), quality, and quantity
	+ at the very least should be reffed in our tree structure
	+ do we want to measure variety the way they do?

## Reiterate Goals, Establish plan of attack, introduce terminology

+ reiterate goals from above at high level


## Method of execution

### Experiment

+ high-level summary of experiment already done; leave it for revision

+ participant stats
	+ number of participants
		+ 341 HITs completed
		+ 280 distinct workers (display stats in table for each question)
		+ explain additional responses for iPod
	+ repeat participants
		+ plot of number of participants over time
		+ implications
	+ completion

+ response stats
	(print in table: question x number of responses, number of non-repeat responses)


+ minimum of 7 responses per condition after data loss do to repeat workers. Table?

### Coding

#### Clustering

+ this part is in the document already

#### Manual comparisons

+ need to describe this, but leave until we actually have data from it

#### Translation to variables of interest

+ originality: category tree o-score
	+ justify this choice
+ specificity: height in the subtree - replace with generality
	+ it would be great to show if this correlates with our other relative specificity score, but that may not be possible
	+ justify height vs depth

+ correlations between variables; are we double-measuring?

## Findings

### Describing the data

#### The coded data

+ instance/idea stats
	+ table: number of ideas, number of unique ( in condition) ideas, % unique responses
	number of instances, number of non-repeat instances

+ tree stats (iPod only)
	+ number of idea nodes
	+ number of category trees
		+ distinguish between singleton tree and trees with... more than one instance (most generous interpretation of the "image" phenomenon from SIAM)
	+ table: number of singleton trees, non-singleton trees, # ideas in each
		+ possible the chart, but I think not
	+ number of unique ideas per tree (histogram)
	+ number of instances per tree (histogram)
	+ tree depth histogram
	+ parent degree histogram
	+ parent nodes artificial (table, all nodes, big category nodes)

#### Measures
+ measures of outcomes of interest ( this should move)
	+ originality: o-score (Design fixation)
	+ specificity: ????

#### Zoomed out

Introduce two ways of looking concept: overall input of instances over a certain number of ideas received.

Show one unshuffled plot, then introduce concept of shuffling.

+ saturation:
	+ number of ideas received
	+ number of category trees received

+ effect of single person phenomenon?

+ originality (hist cross condition?, cumulative originality is a dangerous concept)
+ specificity (hists cross condition?)

#### "Zoomed in"

+ run stats
	+ number of categories normalized(hist)
	+ number of unique ideas (hist)
	+ how long people tend to riff on the same idea (hist)
	+ height range

+ originality
	+ category tree level
	+ idea level

+ introduction of new ideas and categories (where does this happen?)

+ specificity
	+ height in tree

+ fixation
	+ heatmap of remixing only? Or remixing and time spent on idea?
	+ length of fixation runs histogram

+ time spent on responses

### Hypothesis testing

+ base this on what we see int he visualizations, and related work

+ viz-founded
+ in larger conditions, participants are more likely to work in the same categories, and to reiterate the same idea (exhaustion) (how does this reconcile with their ideas being more original?)

### Qualitative observations

+ personas?
+ Miley Cyrus effect
+ willingness to take on larger HITs effect (citation?)

### Recommendations

## Future work

### Interventions

+ facilitation/training
+ those implied by the findings: breaking out of the early "commons"

### Modeling

+ can turkers build the cluster model?
+ identify personas; how long does it take to confidently identify a type of brainstormer?

### ML

+ automatically categorizing ideas or identifying their originality
+ at a simpler level, automatically infer remixing (phrase bouncing)