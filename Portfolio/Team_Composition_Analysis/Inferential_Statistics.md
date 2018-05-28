# Inferential Statistics

> *"Submission: Write a short report (1-2 pages) on the inferential statistics steps you performed and your findings. Check this report into your github and submit a link to it. Eventually, this report can be incorporated into your Milestone report."*

During exploratory data analysis, several forms of inference were performed:

- Feature Correlation Analysis
- Independent t-tests (with equal and unequal $\sigma$)
- Multiple Hypothesis Testing (with Holm-Sidak correction)

---

### Feature Correlation Analysis

Feature correlations were examined visually by plotting their correlation matrix in a seaborn heatmap. This revealed that there were quite a few pockets of strongly correlated variables. This meant the possibility of multicolinearity and was important to note for modeling decisions later on. Either a model robust to multicolinearity would need to be chosen, correlated features would need to be dropped/combined, or a dimensionality reduction would have to be performed.

This lead to the decision to perform Variance Inflation Factor (VIF) testing later in the analysis to identify redundent variables and sources of multicolinearity.

Correlation analysis on features with the target (`result`) also was performed but didn't reveal any unexpected findings.

### Independent t-tests

- **Sneaking Wins From Better Opponents**  

Graphical examination of the data revealed a possible insight -- Low quality teams, even in games they were behind in gold at 15 minutes, had greater success against top competition when they were able to take an early baron. How is unclear (steal? sneak? lucky team fight?), but significance could be confirmed even against the most conservative of $\alpha$'s.

- **Map Side**

There has often been a claim (and was also observed through earlier correlation examination) that blue side wins more often than red. This was put to the test and found to be very statistically correct, with a t-statistic of over 10 standard deviations.

This confirmation began another question: Does blue win more often because of map advantages around baron or pre-game advantages in the draft?

Another strongly correlated feature with `result` was 'fb' (First Blood).  Since first blood generally occurs before baron control has influenced the game, it was a good place to look for an answer. If the blue side outperformed  red in first blood, it would be reasonable to think the advantage came from pregame advantages (picks and bans).

- **First Blood**

This t-test also confirmed that blue had a statistically significant advantage over red in scoring first blood (though not as heavily favored as win rate).

### Multiple Hypothesis Testing

The earlier testing led to a new question: How could the game developers allow such an imbalance to persist? To answer this, data was grouped by patch and side, average wins were then plotted, revealing what appeared to be a massive, consistent advantage for blue side.

However, when those same series of wins/losses were compared with multiple t-tests (using a Holm-Sidak correction), it was determined that while blue side did indeed have an advantage, its advantage was only statistically significant in 18% of the patches, as opposed to the 95+% the graphical interpretation would have suggested.
