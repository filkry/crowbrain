# Main rebuttal content

## M1

> The results of the work are not generalizable because the data came
from only a single task and a single run of that task and there were no
test of generalizability on unseen data. The models presented in the
paper are appreciated and quite interesting, but a significant weakness
is that the authors built the models using only a single brainstorming
question and are trying to make generalizable claims. For example, the
authors state (p. 7), “there is a set of general, common ideas that
make up the first several responses of every crowd brainstorming
session.” How can this claim be made when there was only a single
brainstorming question posed? Technically, the claims in the paper can
only be made for that specific brainstorming question (how to reuse old
mp-3 players). The paper would be far more convincing had the authors
showed that the models fit the data generated from multiple questions.
Alternatively, it could have been acceptable if the authors conducted
additional runs of the same question and showed that the models remained
the same or similar. Claiming generalizability of models trained on only
a single data set is a major limitation of the work. It does not mean the
work is bad, only that a larger data set would greatly strengthen the
work. Even R1, the strongest advocate for the paper, observes that there
is no validation of the model and questions its validity, which is the
main contribution of the work.

It was not our intention to suggest the findings were highly generalizable, but rather to provide evidence in support of the stated hypotheses as well as present models that can be applied to generalized datasets. Examining phenomena in the context of a single problem is not unusual in creativity literature, and the paper cites several examples in psychology (3, 4, 17, 22) and crowd creativity research specifically (28, 29).

In addition, this is the first work to examine crowd creativity in the large scale. Even in the context of a single question, this work explores the diversity of a massive sampling.

REV: change quoted line to refer to limit application to study data explicitly

## M2

> The motivation for the work is very weak. It is not clear why it is
important or interesting to build models of brainstorming of workers on
Mechanical Turk. For example, in prior work, such as on the SIAM model
developed by Diehl and Strobe, it is clear that the research was directed
at testing a particular model of human memory as it relates to
brainstorming. In the current paper, however, it is difficult to
understand for what purpose the authors are constructing the models. Are
you testing a particular theory of human memory? Are you testing a
particular theory of group brainstorming? How do the authors intend for
the models to be used, e.g., in what work contexts and for what purpose?
As implied by R2, it also does not appear that there was anything
specific about the use of mechanical turk for this work and therefore the
authors should offer a deeper comparison to brainstorming in group
contexts.

Crowds are uniquely suited to creativity generation tasks. Brainstorming has been independently introduced as either a component of a task or a major task itself in prior crowd creativity work [16, 29]. In cases where idea generation is a prerequisite to creative task completion, it is desirable to have a set of models and guidelines to inform task design. Finally, it cannot be assumed that findings in group or electronic brainstorming are generalizable to a microtask marketplace context.

## M3

> There were a number of methodology confounds. One serious confound is
that workers were producing different numbers of ideas. For example, in
the “5” condition, 57 people produced 287 ideas; whereas in the
“50” condition, 9 people produced 450 ideas. Given that prior results
on brainstorming show that better ideas come later in the idea generation
phase for an individual, the fact that this was not accounted for in the
modeling is a serious oversight.  An alternative might have been to ask
every worker to produce the same number of ideas, but manipulate the
number of workers asked. This may remove the confound yet still allow for
modeling and interesting outcomes.

The models treat all instances that occur in the same position in a brianstorming run equivalently, across condition. We tested for differences in originality scores between the first 5, 10, 25, etc responses as a function of condition, and found no two conditions significantly different.

Note, it may be more valuable to just run our analysis separately for each condition and see if the hypotheses hold.

REV: add this to the analysis section

# M4 (+ other reviewers)

> How were the off-topic (non-sense) responses filtered from the data
set? The problem with originality is that it rewards off-topic responses.
As noted by R3, the more “original” ideas may also have been the more
off-topic ideas and it is likely that these types of ideas start to
appear after the easy ones run out. Asking one person to come up with 100
sensible responses for using an old MP-3 player might be a stretch. Can
the authors offer evidence that the ideas after the first 20 were both
on-topic and original?

> Listing 100 ideas at one time seems to be a challenging task. Prior work
shows that when a task becomes difficult, workers attempt to give random
answers. In the analysis, the authors only removed the repeated responses
from the workers, but did not remove the off-topic responses. The
analysis shows that requesting more than 20 ideas from each worker can
improve the “originality”. However, another explanation is that when
a worker is required to generate more 20 ideas, s/he is more likely to
produce random and irrelevant responses.

> The researchers described a few example cases where participants had
been extremely creative or made systematic manipulations of the same
idea, which prevented the researchers identifying differences between
their conditions.  How common where these cases?  Why not remove these
subjects from your dataset?

Nonsense responses were flagged under the condition that their application to the iPod problem could not be visualized by the coder. This was not mentioned in the analysis due to their low prevalance (less than 12 instances total). We re-ran all analyses with these responses removed and all hypothesis tests held (with very slight differences in posterior parameters). They were initially left in as representative of "real world" results.
REV: describe nonsense answers in data description and coding process

## M5 (+ other reviewers)

> The authors need to better describe how the o-scores were computed. In
the Jannson and Smith paper, the o-scores were computed from pictorial
representations of the designs. Here, the ideas are in the form of text.
So how were the o-scores computed and how can you interpret the o-score
from the reader’s perspective? Also, as noted by R3, the o-scores
appear to be way off. How can all of the o-scores be between .994 and
.999 (Figure 5).

> The authors very clearly describe early on in the paper how
originality will be measured e.g. o-scores.  However, this leaves some
confusion around how categories of ideas will be measure --- note that
the authors do explain this in the paper, but question if a reference to
how categories of ideas are measured earlier in the paper would help
readership.

> The o-score in the worst condition is still more than 0.994 (see Figure
5), which is very close to the maximum value of o-score. This means that
most of the ideas generated in the conditions are highly original and
originality is not the main issue for this particular brainstorming
question (the reuse of MP3 players). The authors should redesign the
brainstorming question to test the hypotheses.

O-scores were computed as follows. A count of all instances in the idea node (for idea o-score) or category tree (for category o-score) was taken. This was divided by the total number of instances and subtracted from one to arrive at the o-score. While the scores seems unusually clumped, the o-score is a relative measure for distinguishing between ideas dominated by the total number of unique responses. The "squishing" of scores into the .994-.999 range is an artifact of the volume of the data set and representative of the space of possible responses.
REV: clarify the o-score measure and explain the meaning of the response range

## M6 (+ other reviewers)

> The clustering technique did not follow standard protocol, yet the
validity of the models depend on how the clustering was done.  Both R1
and R3 raised this concern. For example, R3 notes that “the clustering
results are only produced by one primary author. The authors need to
follow the standard procedure in qualitative research and the second
researcher should produce the clustering results independently (not just
review the results produced the primary author).”

> I slightly question the process and validity of the inter-judge
reliability test.  It seems as if the second rater was given the
categories and asked to validate them.  A better approach would have been
to give the second rater the ideas belonging to those randomly sampled
category tree, form category trees themselves and then look at
differences.  As the authors describe, the rating does not seem to
validate different instances of the same idea e.g. "Storage Container"
and "Small Storage Box".

> The validity of the findings highly depends on the clustering results.
But the clustering results are only produced by one primary author. The
authors need to follow the standard procedure in qualitative research and
the second researcher should produce the clustering results independently
(not just review the results produced the primary author). Another
possible way is to use clustering methods to get the results
automatically without the interference of the primary author.

Mike: I need your help here.

## M7

> The hypotheses in the paper have no framing and may be better
characterized as research questions. For instance, the first hypothesis
states that the probability that an idea is novel decreases non-linearly
as a function of the number of responses received. Where did this
hypothesis come from? How was it derived? How does it relate to prior
work? Perhaps the hypotheses could be better framed as a set of
interesting research questions, which would be fine by me. The first
question, for example, might be better phrased as “how does the
probability that an idea is novel relate to the number of ideas
received?” This allows a bit more freedom into exploring the various
models as opposed to locking into a specific set of models (e.g.
non-linear models).

We agree that the first and second hypotheses would benefit by being expressed as research questions. The other three hypotheses are informed by prior work.
REV: As suggested.

## M-Extra

> Why are you providing Osborne’s rules of brainstorming to individual
users? As far as I know, those rules were targeted at group brainstorms,
not for individuals.

The rules of brainstorming are often given in an individual context, including in very early comparisons of group vs nominal brainstorming [27].

> The authors should focus on idea quality, rather than quantity or
originality. The problem with originality is that highly ‘original’
responses could be utter non-sense. The authors should report how the
ideas were filtered to remove any irrelevant results and possibly
consider idea quality in future work. 

We chose to emphasize uniqueness as a common component of creativity. Number of unique ideas also correlates with quality overall [7, 4, 22, 21, 25, 6]. We agree that an explicit analysis of quality is imperative future work.
REV: explicitly state choice to emphasize uniqueness, add quality to future work

# Other reviewers

> I was interested as to why the researchers dropped all other
questions than the MP3 question.  Either don't mention it or provide an
explanation.

Not sure what to say here. "We didn't have time!"

> The researchers modeled their data using the full dataset.  Usually,
when creating a model you split your dataset into three.  One used to
create your model, one used to test your model and another used to
validate your model.  As such, there is no validation of this model,
which is one of the main contributions of this paper.  Furthermore, given
the high variance described, I would further question the validity of
this model.

This paper leverages its models specifically to test hypotheses. We do not evaluate the fit or predictive power of the models.


> The authors do not really discuss how this strategy for brainstorming
differs from, or is similar to, group or individual brainstorming not
online. This deserves a substantial discussion and proposal of future
work. 

> The future work and conclusion sections are the weakest part of the
paper. The authors should carefully consider how their results can be
used by other researchers and what clear (and generalizable) findings can
be made. In addition, I challenge the authors to more thoughtfully
consider the future work to drive the future of HCI research in this
field.

TODO: future work revisions

> To compare the originality in condition <= 20 and condition >20 (section
“Originality”, page 7), the authors first used models to fit the data
and then compared parameters of these models. However, some of the models
obviously did not fit the data well (see Figure 8 left bottom). This made
the comparison results inaccurate. Since it is non-normal distributed
data, the authors could perform a non-parametric test (e.g. Mann-Whitney)
to directly compare o-scores in these two conditions. Similar issues are
also present in the section “Most popular Ideas”.

TODO: run this test

> In the section “Most popular Ideas”, the authors compared the
originality in the condition “first 5 instances” and the condition
“last 5 instances”. In the “first 5 instances” condition, the
ideas are the mixture of ideas in all the conditions including condition
5, 10, 20, 50, 75, and 100, while the ideas in the “last 5 instances”
condition are only from the condition 100. I’m not sure this is a fair
comparison. It would be better if the authors performed the analysis
separately in each condition. Also, if a worker needs to submit 100
ideas, it is reasonable to consider that the first 5 ideas are generated
in the early phase. But if a worker is aware of that he only needs to
submit 5 ideas, these 5 ideas are generated from the entire brainstorming
session (NOT the “early” phase of the brainstorming session).

With respect to the test of hypothesis 3, we feel the description of the test was poorly given and led to misunderstanding among reviewers. We did not compare the first 5 instances and last 5 instances for the presence of common ideas, but rather the first 5 instances with the *remaining* instances (all those with position greater than 5). In addition, this was done only for runs with 10 or more responses (conditions 10, 25, 50, 75, 100), to ensure that no participant could contribute to one distribution without contributing to the other.
REV: Rephrase the description of the hypothesis test to be more descriptive

# Maybe Ignore

> The authors have clearly stated the methodology of the paper and have
presented testable hypothesis. Their contribution to the HCI community is
clearly stated (It should be noted that I greatly appreciate the detailed
contribution statements and the bullets at the end of the introduction-
this provides a very clear outline for the paper!) and I feel that it
does add to the community knowledge. I do wish that the authors had
compared their findings from Mechanical turk to a non- distributed team
and compare findings. This type of study would help bridge the gap and
identify differences in these two communities and also aid researchers in
understanding which study findings generalize to both audiences. I think
this deserves some discussion in the paper. 

> The authors only had participants textually describe their ideas -
however they spend little to know time describing why. This is important
as prior research has explored differences in textual and pictorial
brainstorming. Please clarify and provide relevant source work. 

> The authors mention the popular shah and vargas-hernandez method for
rating idea novelty but make little reference to it later on. It should
be better noted why this method was not used to test originality and
popular ideas... Because this is the 'go-to' method, it makes the
findings and comparison of these results to other studies more difficult.

> It would be great if the authors reported more information on the
collected ideas (e.g. the length of the collected ideas). It seems that
the length of example ideas in the paper is much shorter than the median
length of ideas in previous work (e.g. 60 words).

# Textual changes

> The authors provide a lot of details in the data collection section on
why they didn't do certain things, e.g. removed data from the unlimited
condition etc. However, this additional information makes this section
hard to comprehend. It may be a better strategy to just simply say what
you did and why you did it - revise. 

> Title headers are floating at the bottom of page 4, 5, and 6.

> Details of clustering was too brief on page 5 to comprehend how this
was accomplished, "Clustering was performed using a custom-built tool".
Either provide more details or a reference for this strategy. 

> on page 7 the list is split across the two pages. Please fix.

> The first sentence of the abstract is difficult to understand and should
be re-written.

> There are a few places that need to be edited, such as the floating
sentence on page 2, "Finally, Diehl and Stroebe found that groups that
generate good ideas also generate many ideas [7], a ﬁnding that has
been veriﬁed many times [4, 22, 21, 25, 6]." In fact, there are several
weird 2 sentence 'paragraphs' throughout that should be edited. In
addition, the authors mention several places throughout, "we are unaware
of..." This should be edited to be sound more professional and accurate.
When there is something like this written it makes me question if the
authors have completed a substantial review of the literature.

# M8 + others: Related work additions

> Both R1 and R2 list a series of citations that should be included in a
revision of the paper but none of them appear detrimental to the
contribution of the present work. 

> The literature review includes a good mix of work on creativity from
the psychology community and later work on Group Decision Support
Systems.  Some recent research the authors have missed or not included
is:

> Warr, A. and O'Neill, E. (2005) The Effect of Operational Mechanisms on
Creativity in Design. Proc. Interact 2005, Rome, Italy. Springer LNCS
3585, 629-642.

> Warr, A. and O'Neill, E. (2006). The Effect of Group Composition on
Divergent Thinking in an Interaction Design Activity. DIS 2006, 26-28
June 2006, Penn State, USA.

> The above research presents an experiment that examines the effect of
group compositions on creativity in terms of number of generated ideas
and number of categories of ideas, respectively.

> According to the literature, a creative idea is an idea that is both
novel and appropriate.  Where novelty is an indicator of uniqueness and
appropriateness is what makes an idea creative.

> Amabile, T. M. (1983) The Social Psychology of Creativity.
Springer-Verlag, NewvYork.

> Hence, while measuring novelty is a measure of uniqueness, the measure of
appropriateness is what differentiates creative ideas from the bizarre
and odd. 

> Other references recommended for review and citation:

> Furnham, A., & Yazdanpanahi, T. (1995). Personality differences and group
versus individual 
brainstorming. Personality and Individual Differences(19), 73-80. 

> Nijstad, B. A., Stroebe, W., & Lodewijkz, H. F. M. (2002). Cognitive
Stimulation and interference in 
groups: Exposure effects in an idea generation task. Journal of
Experimental Psychology, 
38(2002), 535-544. 

> Nijstad, B.A., Stroebe, W. and Lodewijkx, H.F., 2003, “Production
blocking and idea generation:
Does blocking interfere with cognitive processes”, Journal of
Experimental Social Psychology,
39(6): 531-548.

> Xu, A. and B.P. Bailey. A Crowdsourcing Model for Receiving Design
Critique. Proceedings of the ACM Conference on Human Factors in Computing
Systems, Extended Abstracts, 2011, 1183-1188


# Re-usable praise

> This paper makes a number of contributions:
Demonstrates the application of microtask marketplaces to
brainstorming
Provides models for rates of idea generation, divergence and
uniqueness
Show that the first 20 ideas are less original than those generated
after

> Overall, I would argue for accepting this paper.  IMO, it is one of the
best papers I have read on the study of creativity recently and is
furthermore, very scientific in its application --- something very
difficult in the domain of creativity.

> The researcher conclude by making actionable recommendations based on
data that are applicable to researchers and practitioners.

> I am truly excited by the scope of this research in the future.  For
example, explaining differences between demographics, personality types,
etc... As well as validating the research with different brainstorming
tasks.

> Through a series of experiments, the authors provide a characterization
of brainstorming on a microtask marketplace (Amazon Mechanical Turk), and
provide a model for the rate of idea generation, rate of category
generation and uniqueness of generated ideas through a series of
experiments.  The authors have also identified two clear audiences for
their findings including both individuals who wish to leverage the crowd
for brainstorming and those interested in developing methods for
optimizing the brainstorming process. I feel that these items clearly
contribute to the HCI community but the article needs to be revised
before it is acceptable for publication.

> This paper reports on an investigation of brainstorming in micro-task
marketplace. The topic is very interesting and research hypotheses are
original. I hope this work is published eventually. But the experimental
design and data analysis were flawed.

# Boilerplate

> This paper presents a series of experiments aimed at modeling
brainstorming behavior in crowds (as defined by Mechanical Turk).
Specifically, the authors modeled the rate of idea production, including
originality, to a brainstorming task and offered data-driven insights
into how to best leverage the use of crowds for brainstorming. 

> The reviewers were very much split on this paper. R1 strongly advocates
for acceptance and states that “this one of the best papers I have read
on the study of creativity recently and is furthermore, very scientific
in its application --- something very difficult in the domain of
creativity.” I agree the authors are pursuing an original line of
scientific inquiry and appreciate R1's enthusiasm for the work. R2 also
proposes to accept the paper and believes that the paper adds to the
community knowledge. R3, however, is not sold on the work and offered a
very detailed analysis of the statistical results. From my own reading of
paper, I think this paper could be outstanding but currently has a lot of
weaknesses, some serious, that might require additional work. 

> The paper will likely receive an additional review and discussion at the
program committee meeting. So the authors are kindly asked to direct
their rebuttal at these points:

> There are several points in the paper that need clarification: