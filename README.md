# Efficient Prediction of Region-wide Traffic States in Public Bus Networks using LSTMs


Public bus systems are impacted by many factors, such as varying traffic conditions, passenger demand, and weather changes. One can combine all those factors that affect bus travel times into a single factor called link state, where a link represents part of a bus route. Several works exist that predict single link states using different statistical and machine learning approaches. More recently, deep learning techniques, such as LSTMs, started to be used to predict the state of entire bus routes. The main problem with this approach is that it uses extensive computational resources.

In this work, we evaluate the use of LSTMs to predict the state of entire city regions instead of single routes. It has two advantages: (i) the state of each link is evaluated only once for all the bus routes that cross it, and (ii) information from buses from all routes can be used to determine future link states. Using a shallow bidirectional LSTM architecture produced accurate state predictions with an average MAPE of $12.5$. Moreover, we show that it can be trained daily and used to predict link states in real-time for a large metropolis, like São Paulo.

