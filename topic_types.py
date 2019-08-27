import numpy as np
import pandas

def check_safe_bet(topic_info, years_weights_stats, num_years=None):
    sorted_years = sorted(topic_info["years"].keys(), reverse=True)
    if num_years is not None:
        sorted_years = sorted_years[:num_years]
    for year in sorted_years:
        weight = topic_info["years"][year]
        ymean, ystd = years_weights_stats[year]["mean"], years_weights_stats[year]["std"]
        if weight < (ymean + ystd):
            print("Topic not safe bet due to w {} on year {} with avg + std: {}".format(weight, year, ymean + ystd))
            return False
    return True

def check_hibernating(topic_info, years_weight_stats, num_giant_years=None):
    sorted_years = sorted(topic_info["years"].keys(), reverse=True)
    checking_hibernation = True
    hyears, gyears = [], []
    for year in sorted_years:
        weight = topic_info["years"][year]
        ymean, ystd = years_weights_stats[year]["mean"], years_weights_stats[year]["std"]
        if checking_hibernation:
            if weight < (ymean + ystd):
                hyears.append(year)
            else:
                if not hyears:
                    return False
                checking_hibernation = False
        else:
            if weight > ymean -ystd:
                gyears.append(year)
            else:
                if not gyears:
                    return False
                break
    return (gyears, hyears)


def check_emerging(topic_info, year_w_stats, year_g_stats, policy="any"):
    # check a range with low topic weight wrt year average, if any
    low_weight_years = []
    high_growth_years = []

    sorted_years = sorted(topic_info["years"].keys(), reverse=True)
    for year in sorted_years:
        topic_growth = topic_info["growths"][year]
        year_mean_growth = years_growths_stats[year]['mean']
        year_std_growth = years_growths_stats[year]['std']

        if topic_growth > (year_mean_growth + year_std_growth):
            high_growth_years.append(year)
        else:
            break
    # if no high growth, halt
    if not high_growth_years:
        return None

    for year in sorted_years:
        if year in high_growth_years:
            continue
        topic_weight = topic_info["years"][year]
        year_mean_weight = years_weights_stats[year]['mean']
        year_std_weight = years_weights_stats[year]['std']

        if topic_weight < (year_mean_weight - year_std_weight):
            low_weight_years.append(year)
        else:
            break
    if not low_weight_years:
        return None
    if policy == "full":
        # has to span the entire year
        if not (high_growth_years + low_weight_years == sorted_years):
            return None
    return (low_weight_years, high_growth_years)


x = pandas.read_csv("topic_trends_per_year.csv")
topics = {}
years_weights = {}
years_growths = {}

years_weights_stats = {}
years_growths_stats = {}

topic_avg_w_across_years = {}


# read
for row in x.iterrows():
   topic, topicname, year, weight = row[-1]
   if topic not in topics:
       topics[topic] = {"years":{}}
   topics[topic]["years"][year] = weight
   if year not in years_weights:
      years_weights[year] = []
      years_growths[year] = []
      years_weights_stats[year] = {}
      years_growths_stats[year] = {}

dict_diffs = {}
dict_growth = {}
# calc diffs and averages
for topic in topics:
    years_keys = sorted(map(int, topics[topic]["years"].keys()))
    values = [topics[topic]["years"][y] for y in years_keys]
    diffs = [None] + [latter - former for (latter, former) in zip(values[1:], values[:-1])]
    grs = [None] + [(latter - former)/former*100 for (latter, former) in zip(values[1:], values[:-1])]
    if "diffs" not in topics[topic]:
        topics[topic]['diffs'] = {}
    topics[topic]['diffs'] = {y:d for (y,d) in zip(years_keys,diffs)}
    if "growths" not in topics[topic]:
        topics[topic]['growths'] = {}
    topics[topic]['growths'] = {y:g for (y,g) in zip(years_keys,grs)}

    topic_avg_w_across_years[topic] = np.mean(values)

    for year in years_keys:
       years_weights[year].extend(values)
       years_growths[year].extend(grs)

for year in years_weights:
   w, gr = [x for x in years_weights[year] if x is not None], [x for x in years_growths[year] if x is not None]
   try:
       years_weights_stats[year]["mean"], years_weights_stats[year]["std"] = np.mean(w), np.std(w)
       years_growths_stats[year]["mean"], years_growths_stats[year]["std"] = np.mean(gr), np.std(gr)
   except:
       print(w)
       print(gr)

dcol, gcol = [], []
yavgcol, ystdcol, tavgcol = [], [], []
for row in x.iterrows():
   topicid, topicname, year, weight = row[-1]
   dcol.append(topics[topicid]['diffs'][year])
   gcol.append(topics[topicid]['growths'][year])
   yavgcol.append(years_weights_stats[year]["mean"])
   ystdcol.append(years_weights_stats[year]["std"])
   tavgcol.append(topic_avg_w_across_years[topicid])

x["diffs"] = dcol
x["growth"] = gcol
x["year_avg_w"] = yavgcol
x["year_std_w"] = ystdcol
x["topic_yearwise_avg"] = tavgcol
x.to_csv("expanded_cols.csv")
# print(x)

emergings, safes, hibernating = [], [], []
# emerging
for t in topics:
    topic_info = topics[t]
    topic_avg = topic_avg_w_across_years[t]

    res = check_emerging(topic_info, years_weights_stats, years_growths_stats)
    if res is not None:
        print("Topic {} emerging, times:", res)
        emergings.append(t)
    if check_safe_bet(topic_info, years_weights_stats):
        print("Topic {} is safe.")
        safes.append(t)
    if check_hibernating(topic_info, years_weights_stats):
        print("Topic {} is hibernating.".format(t))
        hibernating.append(t)
print("Got {} emerging topics, {} hibernating and {} safe bets".format(len(emergings), len(hibernating), len(safes)))
