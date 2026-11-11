A modified implementation of Mamba leveraging a state dependent recurrent mechanism 
incorporating Pytorch
-------------------------------------------------------------------------
Gist:
In conventional mamba networks Bt , Ct and Δt are projected directly from xt which is the input vector. In other words, xt controls what to be written in Bt and Ct.  In some cases, the current word is pointing at a preceding word. E.g Consider the following sentences:
"The animal was hungry. It searched everywhere for food. Finally, it found a mouse."
Imagine the model has already read:
"The animal was hungry."
Now it reads "it".
•	Original: (Bt) is determined by the representation of "it". The previous state contributes separately to the state update.
•	Modified: (Bt) is determined by both "it" and the existing memory. Therefore, the model can learn to write "it" differently depending on what it has previously read.
For example, the same word "it" might be processed differently depending on whether the previous context discussed an animal, a car, or a computer.

Initial tests showed slight improvement in PPL. The original and modified models as well as initial test file are available at:
Git repo: https://github.com/amirho3inz1/mamba-state-dependent
Original repo: https://github.com/johnma2006/mamba-minimal

