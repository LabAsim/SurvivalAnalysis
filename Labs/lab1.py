"""Lab 1"""
import os.path
import warnings

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from collections import Counter

from pandas.errors import SettingWithCopyWarning

from helpers import str2bool

# EMPIRICAL SURVIVAL ESTIMATE
# Consider the following data from the Ohio State University Bone Marrow Transplant Unit on
# the time to death or relapse (in months) after bone marrow transplant for 12 patients with
# non-Hodgkin’s lymphoma. For the purposes of this exercise, assume that there are no
# censored event times.

# a. Using the above data, calculate the empirical estimate of the survival function, 𝑆(𝑡+),
# by hand. Summarize your calculations in a table containing the survival times, the
# number of deaths at each time point, and 𝑆(𝑡+). Use your own R code to verify your
# calculations.


time = [1, 2, 2, 2, 3, 5, 6, 7, 8, 16, 17, 34]

# Create a Dataframe with unique values from time
df = pd.DataFrame(data={"time": np.unique(np.array(time))})

# Count the values in time and add them as a column named failures
df["failures"] = df["time"].map(Counter(time))

# Number of patients being alive
df["alive"] = 12 - df["failures"].cumsum()

# Survival probability
df["surv"] = df["alive"] / 12

print(df)
df["event"] = 1
print(df)

from sksurv.nonparametric import kaplan_meier_estimator

nhl = pd.DataFrame(data={"time": time})
nhl["event"] = True
time, survival_prob, conf_int = kaplan_meier_estimator(
    nhl["event"], nhl["time"], conf_type="log-log"
)

nhl_results = pd.DataFrame({"time": time, "prob": survival_prob, "lower CI ": conf_int[0], "upper CI ": conf_int[1]})
print(nhl_results)

plt.step(time, survival_prob, where="post")
# plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
plt.ylim(0, 1)
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
# plt.show()

# (d)
# Calculation of standard errors
# Just the SE of a binomial proportion, p(1-p)/n

df["se"] = pow(base=df["surv"] * (1 - df["surv"]) / 12, exp=0.5)

# e.
# Identify the estimated median survival.
# The median survival time is the smallest time, τ, such that 𝑆(𝜏+) ≤ 0.5, so the estimated
# median is 5.
# Lower quartile (25%): the smallest time (LQ) such that 𝑆(𝐿𝑄+) ≤ 0.75. Since 𝑆(2+) =
# 0.667, the estimated 25%-ile time is 2.
# Upper quartile (75%): the smallest time (UQ) such that 𝑆(𝑈𝑄+) ≤ 0.25. Since 𝑆(8+) =
# 0.250, the estimated 75%-ile time is 8. To get the above information in R, use the quantile
# function

np.percentile(df["surv"], 0.5)

# KAPLAN-MEIER SURVIVAL ESTIMATE
# We will now use the same data from the Ohio State University Bone Marrow Transplant Unit
# on the time to death or relapse (in months) after bone marrow transplant for 12 patients with
# non-Hodgkin’s lymphoma. However, now the event times that were actually censored are
# noted by “+”.
# 1, 2, 2, 2+, 3, 5, 6, 7+, 8, 16+, 17, 34 +

# Load dataset
parent_folder = os.path.dirname(os.path.dirname(__file__))
nhl1 = pd.read_csv(os.path.join(parent_folder, "R labs/Lab 1/nhl1.csv"))

# The variable nhltime denotes the event time while fail denotes the failure indicator. Note
# that, since we are taking censoring into account, the failure indicator will be zero for the
# censored event times. We tell R that we have survival data through the function Surv. The
# first argument should be the survival time while the second one should be the failure
# indicator.
# Then, survfit can be used to get KM estimates. Note that by using the option data = nhl1
# we tell R where to look for the variables involved in the formula. Add censored = T in the
# summary function if you would like to include the censoring times in the output.


# KM estimates

# Convert 0,1 to False, True. It is needed in kaplan_meier_estimator()
nhl1["fail_bool"] = nhl1["fail"].apply(str2bool)

time, survival_prob, conf_int = kaplan_meier_estimator(
    nhl1["fail_bool"], nhl1["nhltime"], conf_type="log-log"
)

nhl_results = pd.DataFrame(
    {
        "time": time,
        "n_risk": None,
        "events": None,
        "censored_events": None,
        "prob": survival_prob,
        "lower CI ": conf_int[0],
        "upper CI ": conf_int[1]
    }
)
print(nhl_results)

# nhl_results["events"] = nhl_results["time"].map(lambda x: Counter(nhl1["nhltime"]) if nhl1["nhltime"] == 1)
nhl_results["events"] = nhl_results.time[nhl1["fail"] == 1].map(Counter(nhl1["nhltime"]))

# Calculate number of events (without censored) (n.events in R summary of survfit)
nhl_results["events"] = nhl_results.time.map(Counter(nhl1["nhltime"][nhl1["fail"] == 1]))
# The censored ones
nhl_results["censored_events"] = nhl_results.time.map(Counter(nhl1["nhltime"][nhl1["fail"] == 0]))

# Number at risk
nhl_results["n_risk"] = 12
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
warnings.filterwarnings(action="ignore", category=SettingWithCopyWarning)
for ind in df.index[1:]:
    nhl_results["n_risk"][ind] = nhl_results["n_risk"][ind-1] \
                                 - nhl_results["events"][ind - 1] \
                                 - nhl_results["censored_events"][ind - 1]
    # Alternative syntax
    # nhl_results.loc["n_risk", ind] = nhl_results.loc[:, "n_risk"].iloc[ind - 1] \
    #                                          - nhl_results.loc[:, "events"].iloc[ind - 1] \
    #                                          - nhl_results.loc[:, "censored_events"].iloc[ind - 1]


km = pd.DataFrame(
    data={
        "time": nhl_results["time"],
        "dj": nhl_results["events"],
        "rj": nhl_results["n_risk"]
    }
)

km["lambda_hat"] = km["dj"] / km["rj"]

# km["surv"] =
