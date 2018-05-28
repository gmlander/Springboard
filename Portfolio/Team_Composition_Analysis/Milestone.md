# Team Compositions in Professional League of Legends
***

The data being used was obtained from [Oracle's Elixer] (using upload from September 18th, 2017). A [codebook] for the variables is also included.

- **Part 1: Data Cleaning** -- can be found [here](http://nbviewer.jupyter.org/github/gmlander/Springboard/blob/master/Portfolio/Team_Composition_Analysis/I_Data_Cleaning.ipynb) or at the raw [github version](https://github.com/gmlander/Springboard/blob/master/Portfolio/Team_Composition_Analysis/I_Data_Cleaning.ipynb).
- **Part 2: Exploratory Data Analysis** -- can be found [here](http://nbviewer.jupyter.org/github/gmlander/Springboard/blob/master/Portfolio/Team_Composition_Analysis/II_Exploratory_Data_Analysis.ipynb) or at the [github version](https://github.com/gmlander/Springboard/blob/master/Portfolio/Team_Composition_Analysis/II_Exploratory_Data_Analysis.ipynb)

[Oracle's Elixer]: http://oracleselixir.com/match-data/
[codebook]: http://oracleselixir.com/match-data/match-data-dictionary/

---


# Milestone Report

### Problem Definition and Intended Audience

This analysis evaluates game data from professional level _League of Legends_ (LoL) matches to find prredictors, patterns, and models that may aid a franchise in making pregame strategy decisions in regards to style of play and team composition, both generally and opponent-specifically. As the intended audience is a professional team, this analysis does not make much effort in explaining endemic terminology of the game. For a primer see [here](http://www.espn.com/esports/story/_/id/14545779/guide-league-legends) or [here](https://www.riftherald.com/2016/9/29/13027318/lol-guide-how-to-watch-play-intro).

The main impediment to analyzing League, or eSports in general, in the same manner as a traditional sport, is that it isn't one. Obviously. But it's an important thing to remember. Athletes can’t change their physical strengths and weaknesses from game to game, but League players can and do (or at least their champion's). So how does one predict the effectiveness of a team when its compositional elements are indefinite? That's the question this analysis hopes to answer, through machine learning techniques to bucketize and map the many different observed champion-playstyle combinations into a few discrete, interprettable archetypes that can be used as easily inform drafting/banning decisions.

---

## Data Wrangling

### Data Acquisition

The data with permission of the host was obtained from [Oracle's Elixer](http://oracleselixir.com/match-data/) (using upload from September 18th). Data was hosted in several .xlsx files, divided by season, which were read in directly through pandas' read_excel method from online link and appended together to form one dataset. Because the host often changes features and feature names during regular updates of the dataset, I have relied on a .csv I wrote of the initial data used.

---

### Cleaning 

Rows contained two different types of observations - Game data for players and game data for teams. To fix this I partitioned data into team and player specific dataframes, and to account for the high variability in game data between positions, I further split the player dataframe into 5 more subframes.

Most of the actual cleaning was pretty limited. Certain features required datatype changes - numeric categoricals like patch number and gameid. Excel's infamous datetime format conversion reared its ugly head and I was unable to clean it - the dates were corrupted even in the raw excel files, likely due to how excel read them through the initial json parsing from Riot's API.

Luckily, the data had other features that proxied for date - patch number sequential increases over time, while `split`, `week`, and `game` can be used to form a mulitlevel index that would allow for clear ordering of when games occurred. I didn't implement this because I wasn't performing any time-series analysis and only needed patch number to explain differences in the data at different times.

**Note:** In a future study or augmentation of this one, I may use that technique to explain how playstyles change over the course of a patch.

The only other cleaning procedure needed was a small set of observations where `gameid` was not a unique identifier for games as it should have been. This is likely due to the data coming from multiple Riot API servers and some of them having inconsistency in how they assigned gameid. This was fixed through grouping by gameid to select the duplicate id's, and using unique descriptive features to split the duplicate id's into separate games, and appending a 'b' to the end of the duplicates.

The other worry when I first discovered this was that if there were different games sharing the same gameid, the opposite could also be true - the same games being reported multiple times under different gameid's. I tested for this with the folllowing:
```
test = teamDF.groupby(['league','split','week', 'game','team','totalgold']).gameid.count()
test[test>1]
```
By grouping the data on categorical descriptors (league, split, week, game, team) as well as an integer feature (totalgold) and counting the number of gameid's, I was able to see if the same game existed as two observations. Fortunately, none of the counts were above  1.

---

### Dealing With Missing Values
$\quad$ *Everyone's favorite part of data science!*

Initially I had a simplistic approach to missing values. I built a function to impute by mean or median depending on how outlier heavy the feature was. The function identified anything more than three standard deviations from the feature's mean as an outlier and if a feature was made up of more than 1% outliers, median imputation was used, otherwise mean.

This was an okay methodology. But I knew I could do better. I built a function that would generate a referrence dataframe to explain missing values by feature. The dataframe returned had columns representing the number of observed values, missing values, and their respective percentages.

I used this table as well as the `missingno` library to recognize where the missing values were coming from and what relationship they had to other features. The LPL league data was especially problematic, having missing value %'s near or above 50 in most of its numeric features. I decided to drop this data entirely, instead using it as an additional holdout set for validation later on.

Other features I chose to drop for having near 50% mising values on the rest of the data. There was a major change to the game's design in the middle of the data I was working with - The neurtral dragon objective that gave players cumulative bonuses for killing it was changed to give different bonuses depending on the type of dragon that spawned, as well as a late-game elder dragon that would temporarily augment those bonuses. This is a major part of the game and has a huge influence on how teams strategize to control different sections of the map. Half my data was for this new dragon system, and half for the old. Unfortunately, there's no way to compare the importance of this object (and as a result the variability in its feature space) accross the two different implementations. As a result, I chose to drop the dragon related features entirely. In the future, I will redo my analysis when more data is available to only cover games with the new dragon system.

As for actually imputing missing values, I knew I could do better than simple mean/median fill. My examination of the shape of the missing values suggested that they were Missing at Random (MAR) rather than completely at random (MCAR) - meaning there was a pattern to why and where they were missing. I knew there were a lot of advanced imputation options for MAR data such as MICE, knn, and SoftImpute to name a few.

In search of a viable solution, I constructed a function to test the quality of an imputation method. The function took a scaled dataframe and method as a required argument, sampled the rows without missing values and spiked missing values in proportionate to their missing percentage in the full dataframe. This dummy data was then imputed with the provided method and its result was compared against the true data by root mean-squared error (RMSE). This process was performed across a given number of iterations (default 20) and the final return of the function was a list of RMSE values for each iteration.

Through this imputation validation function I was able to determine that KNN(k=25) was the best method for my data, as it gave one of the lowest RMSE results with extremely low variation between the RMSE's. The result was over a 33% reduction in estimated imputation error from my original mean/median method.

**Note:** This validation function included options to ignore binary variables in the entirely data, include binary variables in modeling but ignore their imputations from the RMSE calculation, or include them in both. There wasn't significant reduction in RMSE on my chosen method for any of these options (though there was some). I decided to make things easier on myself and impute both binary and continuous variables with the same method, then round the binaries up to 1 or down to 0.

---

## Future Data/Technology Sources

This [repo](https://github.com/farzaa/DeepLeague) was brought to my attention shortly after I began this project and I think it would be an excellent tool, not only for the pretrained classifier that could be used to create features likes mobility and proximity, but also for the Riot API tools it comes with. Additionally, a much more advanced future study could be done with my a combination of this project and my positional archtypes evaluate optimal moves for team compositions and relative to game states.

---

## Initial Findings

### Inferential Statistics

During exploratory data analysis, several forms of inference were performed that revealed some useful early discoveries:

- **Sneaking Wins From Better Opponents**  

Graphical examination of the data revealed a possible insight -- Low quality teams, even in games they were behind in gold at 15 minutes, had greater success against top competition when they were able to take an early baron. How is unclear (steal? sneak? lucky team fight?), but significance could be confirmed by independent t-test, even against the most conservative of $\alpha$'s.

- **Map Side Matters**

There has often been a claim (and was also observed through earlier correlation examination) that blue side wins more often than red. This was put to the test and found to be very statistically correct, with a t-statistic of over 10 standard deviations.

This confirmation began another question: Does blue win more often because of map advantages around baron or pre-game advantages in the draft?

Another strongly correlated feature with `result` was 'fb' (First Blood).  Since first blood generally occurs before baron control has influenced the game, it was a good place to look for an answer. If the blue side outperformed  red in first blood, it would be reasonable to think the advantage came from pregame advantages (picks and bans).

- **First Blood**

This t-test also confirmed that blue had a statistically significant advantage over red in scoring first blood (though not as heavily favored as win rate).

- **Map Side Doesn't _Completely_ Matter**

The earlier testing led to a new question: How could the game developers allow such an imbalance to persist? To answer this, data was grouped by patch and side, average wins were then plotted, revealing what appeared to be a massive, consistent advantage for blue side.

However, when those same series of wins/losses were compared with multiple t-tests (using a Holm-Sidak correction), it was determined that while blue side did indeed have an advantage, its advantage was only statistically significant in 18% of the patches, as opposed to the 95+% the graphical interpretation would have suggested.

## Baseline Modeling

After reducing multicolinearity in the data through VIF testing, and standard scaling continuous features, I was able to build an out-of-the-box logistic regression with 99% win prediction accuracy. Unfortunately, that was when using in-game data (and apparently it's not as useful to know if you're going to win when you're already winning).

When I removed all in-game features, that accuracy dropped to 59%. When I further removed features that conveyed team quality, that accuracy further dropped to 56%.

Now the real work begins.
