## Machine Learning & Neural Networks
### (a)
1. Momentum is the rolling average of the past calculated gradients. In each step, the past graidents will still contribute to the update, and current gradient will not contribute to the current update so much as the original SGD update. The effect of current gradient will be gradually added to the update in later steps.  
2. Those model parameters with smaller gradients will get larger updates. This could increase the learning of those model parameters, by balancing the learning rate of each parameter.

### (b) 
During training, dropout randomly set s units in the hidden lay er h to zero with probability `$p_{drop}$` (dropping different units each minibatch), and then multiplies h by a constant `$\gamma$`. We can write this as
```math
h_{drop} = \gamma d \circ h
```
d is a mask vector where each entry is 0 with probability `$p_{drop}$` and 1 with probability  `$(1-p_{drop})$`. `$\gamma$` is chosen such that the expected value of `$h_{drop}$` is h.
```math
E_{p_{drop}}[h_{drop}]_i = h_i
```
1. 
```math
E_{p_{drop}}[h_{drop}] = (1-p_{drop})\gamma h=h
\Rightarrow
\gamma = \frac{1}{1-p_{drop}}
```
2.During training, we would like to alleviate overfitt, and applying drop out is like training multiple simpler models. During evaluation, we want to use all the simpler models together, so that we do not apply dropout.


## Neural Transition-Based Dependency Parsing
(a)  
[ROOT] [I, parsed, this, sentence, correctly]   NA  Initial Configuration  
[ROOT, I] [parsed, this, sentence, correctly]    NA  SHIFT  
[ROOT, I, parsed] [this, sentence, correctly]    NA  SHIFT  
[ROOT, parsed] [this, sentence, correctly]  parsed->I LEFT-ARC   
[ROOT, parsed, this] [sentence, correctly]    NA  SHIFT  
[ROOT, parsed, this, sentence] [correctly]    NA  SHIFT  
[ROOT, parsed, sentence] [correctly] sentence->this LEFT-ARC  
[ROOT, parsed] [correctly] parsed->sentence LEFT-ARC  
[ROOT, parsed, correctly] [] NA SHIFT  
[ROOT, parsed] [] parsed->correctly RIGHT-ARC  

(b) A sentence containing n words will be parsed in `$2n$` steps. Because each word will cost 2 steps, one for moving the word from buffer to stack and the other for deleting the word from the stack, except for the last word in the stack since the last word will not be popped out of the stack. We will still need one step for initial configuration. Hence, the total steps will be `$2n-1+1=2n$`.

(f) We'd like to look at example dependency parses and understand where parsers like ours might be wrong.  
i. Error type: Verb Phrase Attachment Error; Incorrect dependency: wedding -> fearing; Correct dependency: heading -> fearing  
ii. Coordination Attachment Error; Incorrect dependency: makes-> rescue; Correct dependency: rush -> rescue  
iii. Modifier Attachment Error; Incorrect dependency: named -> Midland; Correct dependency: guy -> Midland
iv. Prepositional Phrase Attachment Error; Incorrect dependency: one -> been; Correct dependency: been -> one