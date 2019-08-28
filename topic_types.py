import sys

import numpy as np
import pandas


def check_safe_bet(topic_info, years_weights_stats, num_years=10):
    sorted_years = sorted(topic_info["years"].keys(), reverse=True)
    if num_years is not None:
        sorted_years = sorted_years[:num_years]
    for year in sorted_years:
        weight = topic_info["years"][year]
        ymean, ystd = years_weights_stats[year]["mean"], years_weights_stats[year]["std"]
        if weight < (ymean + ystd):
            # print("Topic not safe bet due to w {} on year {} with avg + std: {}".format(weight, year, ymean + ystd))
            return False
    return True

def check_hibernating(topic_info, years_weights_stats, min_g_years=None, policy="full"):
    sorted_years = sorted(topic_info["years"].keys(), reverse=True)
    checking_hibernation = True
    hyears, gyears = [], []
    #
    # below just for visualization / inverstigation
    hl = ["" for _ in sorted_years]
    weights = [topic_info["years"][year] for year in sorted_years]
    ymeans = [years_weights_stats[year]["mean"] for year in sorted_years]
    ystds = [years_weights_stats[year]["std"] for year in sorted_years]
    reduced_means = [m-s for (m,s) in zip(ymeans, ystds)]
    increased_means = [m+s for (m,s) in zip(ymeans, ystds)]
    boolean_hibernate = [w < incr for (w,incr) in zip(weights,increased_means)]
    boolean_giant = [w > red for (w,red) in zip(weights,reduced_means)]
    # above just for visualization / inverstigation

    index = 0
    while index < len(sorted_years):
        year = sorted_years[index]
        weight = topic_info["years"][year]
        ymean, ystd = years_weights_stats[year]["mean"], years_weights_stats[year]["std"]
        if checking_hibernation:
            if weight < (ymean + ystd):
                hyears.append(year)
                hl[index] = "H"
            else:
                if not hyears:
                    return False
                checking_hibernation = False
                index -= 1
        else:
            if weight > (ymean - ystd):
                gyears.append(year)
                hl[index] = "L"
            else:
                if not gyears:
                    return None
                break
        index += 1
    if not gyears:
        return None
    if min_g_years is not None and len(gyears) < min_g_years:
        return None
    if policy == "full":
        if (hyears + gyears) != sorted_years:
            return None
    return (gyears, hyears)

def check_emerging(topic_info, years_w_stats, years_g_stats, num_low_weight=None):
    # check a range with low topic weight wrt year average, if any
    low_weight_years = []
    high_growth_years = []

    sorted_years = sorted(topic_info["years"].keys(), reverse=True)
    for year in sorted_years:
        topic_growth = topic_info["growths"][year]
        year_mean_growth = years_g_stats[year]['mean']
        year_std_growth = years_g_stats[year]['std']

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
        year_mean_weight = years_w_stats[year]['mean']
        year_std_weight = years_w_stats[year]['std']

        if topic_weight < (year_mean_weight - year_std_weight):
            low_weight_years.append(year)
        else:
            break
    if not low_weight_years:
        return None
    if num_low_weight is None:
        # has to span the entire year
        if not (high_growth_years + low_weight_years) == sorted_years:
            return None
    else:
        # low weight years has to be enough
        if len(low_weight_years) < num_low_weight:
            return None
    return low_weight_years, high_growth_years

def main():
    # expects a csv with:
    # topicid 	topicname 	year 	weight
    try:
        infile = sys.argv[1]
    except:
        infile = "topic_trends_per_year.csv"

    x = pandas.read_csv(infile)
    x = x[x["year"] < 2020]
    topics = {}
    years_weights = {}
    years_growths = {}

    years_weights_stats = {}
    years_growths_stats = {}

    topic_avg_w_across_years = {}


    # read
    for row in x.iterrows():
       topic, topicname, year, weight = row[-1]
       year = int(year)
       if topic not in topics:
           topics[topic] = {"years":{}}
       topics[topic]["label"] = topicname
       topics[topic]["years"][year] = weight
       if year not in years_weights:
          years_weights[year] = []
          years_growths[year] = []
          years_weights_stats[year] = {}
          years_growths_stats[year] = {}

    # calc diffs and averages
    for topic in topics:
        years_keys = sorted(map(int, topics[topic]["years"].keys()))
        values = [topics[topic]["years"][y] for y in years_keys]
        diffs = [None] + [latter - former for (latter, former) in zip(values[1:], values[:-1])]
        grs = [None] + [(latter - former)/former*100 for (latter, former) in zip(values[1:], values[:-1])]
        topics[topic]['diffs'] = {y:d for (y,d) in zip(years_keys,diffs)}
        topics[topic]['growths'] = {y:g for (y,g) in zip(years_keys,grs)}


        # topic average weight
        topic_avg_w_across_years[topic] = np.mean(values)
        # topic weight / growth accumulation
        for year in years_keys:
           years_weights[year].append(topics[topic]["years"][year])
           gr = topics[topic]["growths"][year]
           if gr is not None:
               years_growths[year].append(gr)

    # calculate year-wise statistics
    for year in years_weights:
       # ensure non-nan
       w, gr = [x for x in years_weights[year] if x is not None], [x for x in years_growths[year] if x is not None]

       years_weights_stats[year]["mean"], years_weights_stats[year]["std"] = np.mean(w) if w else None, np.std(w) if w else None
       years_growths_stats[year]["mean"], years_growths_stats[year]["std"] = np.mean(gr) if gr else None, np.std(gr) if gr else None


    # decide topic types
    emergings, safes, hibernating = [], [], []

    for t in topics:
        topic_info = topics[t]
        topic_info["types"] = []

        num_low_weight = None
        res = check_emerging(topic_info, years_weights_stats, years_growths_stats, num_low_weight)
        if res is not None:
            emergings.append(t)
            topic_info["types"].append("E")
        num_safe_years = 10
        if check_safe_bet(topic_info, years_weights_stats, num_years=num_safe_years):
            safes.append(t)
            topic_info["types"].append("S")

        min_g_years = None
        policy="full"
        if check_hibernating(topic_info, years_weights_stats, min_g_years,policy):
            hibernating.append(t)
            topic_info["types"].append("H")

    print("Emerging: num low weight years {}".format(num_low_weight))
    for i, e in enumerate(emergings):
        print("{}/{} : {} {}".format(i+1, len(emergings), e, topics[e]["label"]))
        if i > 8:
            print("...")
            break
    print("Safe: num safe years: {}".format(num_safe_years))
    for i, e in enumerate(safes):
        print("{}/{} : {} {}".format(i+1, len(safes), e, topics[e]["label"]))
        if i > 8:
            print("...")
            break
    print("Hibernating: min giant years: {}, span policy: {}".format(min_g_years, policy))
    for i, e in enumerate(hibernating):
        print("{}/{} : {} {}".format(i+1, len(hibernating), e, topics[e]["label"]))
        if i > 8:
            print("...")
            break
    print("Got {} emerging topics, {} hibernating: and {} safe bets".format(
        len(emergings), len(hibernating), len(safes)))



    # expand original df
    dcol, gcol = [], []
    yavgcol, ystdcol, tavgcol = [], [], []
    yavggrcol, ystdgrcol = [], []
    emcol, hibcol, safecol = [], [], []
    for row in x.iterrows():
       topicid, topicname, year, weight = row[-1]
       dcol.append(topics[topicid]['diffs'][year])
       gcol.append(topics[topicid]['growths'][year])
       yavgcol.append(years_weights_stats[year]["mean"])
       ystdcol.append(years_weights_stats[year]["std"])
       yavggrcol.append(years_growths_stats[year]["mean"])
       ystdgrcol.append(years_growths_stats[year]["std"])
       tavgcol.append(topic_avg_w_across_years[topicid])
       emcol.append(1 if "E" in topics[topicid]["types"] else 0)
       hibcol.append(1 if "H" in topics[topicid]["types"] else 0)
       safecol.append(1 if "S" in topics[topicid]["types"] else 0)

    x["diffs"] = dcol
    x["growth_pcnt"] = gcol
    x["year_avg_w"] = yavgcol
    x["year_std_w"] = ystdcol
    x["year_avg_gr"] = yavggrcol
    x["year_std_gr"] = ystdgrcol
    x["topic_yearwise_avg"] = tavgcol
    x["hibernating"] = hibcol
    x["emerging"] = emcol
    x["safebet"] = safecol

    x.to_csv("expanded_cols.csv")
#    print("\nmax:\n", x.max())
#    print("\nmin:\n", x.min())
#    print("\nsum:\n", x.sum())


    # write one without the per-year information
    y = x[["topicid","topicname","hibernating","emerging","safebet"]].drop_duplicates()
    y.to_csv("expaned_cols_onlytypes.csv", index=None)

if __name__ == '__main__':
    main()
