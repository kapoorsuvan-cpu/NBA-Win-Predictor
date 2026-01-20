# NBA-Win-Predictor
NBA win predictor using linear regression and multiple factors to predict a team's wins in the 2025-26 NBA season.

This project uses a linear regression model in Python to predict NBA team win totals based on three carefully chosen factors: previous-season win percentage, coach continuity, and roster talent. Previous win percentage serves as the baseline indicator of team strength, since NBA teams tend to show strong year-to-year performance persistence due to returning cores and organizational stability. Coach continuity is modeled as a simple binary variable (same coach or new coach) to account for the impact of system familiarity and stability. Roster talent is represented using a custom point system based on All-NBA, All-Star, and All-Defensive selections from the prior season, providing a structured way to capture the influence of high-impact players on team success. The model is trained on recent seasons, converts predicted win percentage into projected wins over an 82-game season, and includes a visualization tool to compare historical trends with new team predictions.

2025-26 Predictions:

<img width="837" height="668" alt="Screenshot 2026-01-20 at 1 42 53 PM" src="https://github.com/user-attachments/assets/dec90305-c7f7-41fb-824f-ee484086c3cb" />



<img width="805" height="594" alt="Screenshot 2026-01-20 at 1 42 00 PM" src="https://github.com/user-attachments/assets/5a3b9696-9192-4012-b567-5a7eb7673f15" />

To visualize, I chose a scatter plot that plots team’s previous season win % with the win % from the next year. I selected previous win % for the x-axis because it is being used as a predictor for future success. The y-axis represents win % for the year after. The data points of previous team’s seasons are plotted as regular points, while the predicted points for the Kings, Heat, and Grizzlies (example teams) are highlighted to show how the predictions I made with my model compare with real, historical data. I also chose to use a single scatter plot rather than multiple dimensions to avoid clutter and to make it easy to understand. The new team predictions were highlighted so they could be compared to historical teams. This allowed me to visually validate if the model’s predictions were reasonable. In the scatter plot, you can see that the highlighted teams could be compared to clusters of historical teams with similar previous win percentages. At the top, I displayed the correlation coefficient, r, to see how strong the relationship between just previous year’s win % and current year’s win % are
