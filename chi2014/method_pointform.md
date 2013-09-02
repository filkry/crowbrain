## experimental design
+ recruited ### participants
+ between subjects or between subjects - some participants are in multiple conditions
+ 24 conditions, 4 question x 6 nr (5, 10, 20, 50, 75, 100)
+ dropped data where participants were exposed to the same question condition twice due to sofware error (rephrase this to be more positive)

In compliance of ethics, participants were informed in the consent process that they could leave the task at any time. This is unlike a real-world setting.

Do we mention the old unlimited condition? Probably. Drop it in data analysis further on.

## recruitment
Sampling from a representative population, as we are making inferences about those engaged in crowd marketplaces.

Participants volunteered for each of the request conditions randomly. This results in self-selection bias across those conditions. However, because we are interested in evaluating the realistic differences between results in real-world conditions that include self-selection, this is maybe not so bad.

## task description
Give consent first, then task. First, introduced to the four rules of brainstorming (cite), paraphrased for applicability in a nominal brainstorming context. This is a minimal attempt at introducing the training requirement of brainstorming (cite).

Following this, the randomly-assigned question was presented. Vertically-aligned series of default-sized single line text inputs. At the end of this list of inputs, a larger input allowed participants to optionally provide any further ideas.


## pilots

Brief overview of piloting process and how we ended up at final set of questions.

+ Do we want to talk about our motivations in tweaking our questions?
	+ to be questions requiring creativity
	+ questions turkers were capable of answering creatively
		+ Osborne reference to expertise?
	+ questions with a conceivable utility metric

## data coding

The data was coded according to a hierarchical clustering scheme, in which all of the ideas received for a single question are organized in a tree structure. A single node in the tree is a cluster which represents a particular solution to the problem. This node/cluster may or may not contain any ideas. All ideas with a common ancestor in the tree belong to the cluster at that common ancestor.

Definitions:

> *affinity*: An idea has high affinity to a cluster if the key problem-related concept behind that idea is the same as the key problem-related concept of the cluster. Affinity is commutative.

> *coverage*: An idea has coverage of a cluster if it is suitably abstract to be representative of all members of that cluster or any child clusters. An idea has no coverage of a cluster it is unrelated to. Coverage is not commutative. Two ideas a, b, have *symmetrical coverage* if coverage(a, b) = coverage(b, a).

Clustering alg'm:

	for each idea:
		idea_node = new cluster with idea as member
		current_node = root
		do:
			best_match = max_affinity(idea_node, current_node.children)

			if best_match.affinity is low or current_node has no children:
				insert idea_node under current_node
				exit do
			else:
				if symmetrical_coverage(idea_node, best_match) and high:
					merge idea_node, best_match
					exit do
				else symmetrical_coverage(idea_node, best_match) and low:
				   	new_parent = new child node of current_node
					insert best_match, idea_node under new_parent
					exit do
				else if coverage(idea_node, best_match) > coverage(best_match, idea_node):
					replace best_match with idea_node in tree
					current_node = idea_node
					idea_node = best_match
				else:
					current_node = best_match

Do we comment on the manual coding of limited data? Utility/realisticness especially?
