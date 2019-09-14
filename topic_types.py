import sys

import matplotlib.pylab as plt
import numpy as np
import pandas
import seaborn as sns


def check_safe_bet(topic_info, years_weights_stats, num_years=None, stdev_w_coeff=1):
    sorted_years = sorted(topic_info["years"].keys(), reverse=True)
    if num_years is not None:
        sorted_years = sorted_years[:num_years]
    thresh = []
    years = []
    for year in sorted_years:
        weight = topic_info["years"][year]
        ymean, ystd = years_weights_stats[year]["mean"], years_weights_stats[year]["std"]
        ystd = ystd * stdev_w_coeff
        thresh.append(ymean + ystd)
        years.append(year)
        if weight < (ymean + ystd):
            # print("Topic not safe bet due to w {} on year {} with avg + std: {}".format(weight, year, ymean + ystd))
            return False
    visualize_safe(topic_info, thresh, years)
    return True




def check_hibernating(topic_info, years_weights_stats, min_g_years=None, policy="full", stdev_w_coeff=1, valid_range=None, do_visualize=False):
    # find years where only one of the conditions exist
    # then find continuous sequences
    giant_threshold, hib_threshold = {}, {}
    sorted_years = sorted(topic_info["years"].keys(), reverse=True)
    year_means, year_flucts = {}, {}

    for year in sorted_years:
        weight = topic_info["years"][year]
        ymean, ystd = years_weights_stats[year]["mean"], years_weights_stats[year]["std"]
        ystd = ystd * stdev_w_coeff
        fluctuation_term = ystd
        # giant_threshold[year], hib_threshold[year] = ymean - fluctuation_term, ymean + fluctuation_term
        giant_threshold[year], hib_threshold[year] = ymean + fluctuation_term, ymean - fluctuation_term
        year_means[year] = ymean
        year_flucts[year] = fluctuation_term

    if any(x<0 for x in giant_threshold.values()):
        print("Negative value in giant threshold{} both as giant and hibernating. Curr stdev coeff is {}".format(year, stdev_w_coeff))
        exit(1)

    gyears, hyears = [], []
    for year in sorted_years:
        weight = topic_info["years"][year]
        if weight > giant_threshold[year]:
            gyears.append(year)
        if weight < hib_threshold[year]:
            hyears.append(year)

    # if both giant and hibernating, the stdw coeff needs tuning
    for year in sorted_years:
        if year in gyears and year in hyears:
            # remove year in both categories
            gyears.remove(year)
            hyears.remove(year)

    gyears.reverse()
    hyears.reverse()
    if not gyears or not hyears:
        return None

    # check "edge regions", make sure that latest is hibernating and earliest is giant
    if not(min(gyears) < max(hyears)):
        return None

    if valid_range is None or (gyears[-1] in valid_range or hyears[0] in valid_range):
        # good!
        if do_visualize is not None:
            visualize_hibernating(hyears, gyears, giant_threshold, hib_threshold, topic_info, output=do_visualize)
        return gyears, hyears
    return None

def check_hibernating_flawed(topic_info, years_weights_stats, min_g_years=None, policy="full", stdev_w_coeff=1):
    """This function does not check for overlapping conditions -- it's crap"""
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
    # consideration "limits" per year
    giant_threshold, hib_threshold = {}, {}
    while index < len(sorted_years):
        year = sorted_years[index]
        weight = topic_info["years"][year]
        ymean, ystd = years_weights_stats[year]["mean"], years_weights_stats[year]["std"]
        ystd = ystd * stdev_w_coeff
        giant_threshold[year], hib_threshold[year] = ymean - ystd, ymean + ystd

        if checking_hibernation:
            if weight < (ymean + ystd):
                hyears.append(year)
                hl[index] = "H"
            else:
                if not hyears:
                    return None
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
    gyears.reverse()
    hyears.reverse()
    visualize_hibernating(hyears, gyears, giant_threshold, hib_threshold,  topic_info)
    return (gyears, hyears)

def check_emerging(topic_info, years_w_stats, years_g_stats, num_low_weight=None, min_num_high_growth=None, stdev_w_coeff=1, stdev_gr_coeff=1, consecutive_growth_years=False):
    # check a range with low topic weight wrt year average, if any
    low_weight_years = []
    high_growth_years = []

    sorted_years = sorted(topic_info["years"].keys(), reverse=True)

    lw_thresh, hg_thresh = {}, {}
    for year in sorted_years:
        year_mean_weight = years_w_stats[year]['mean']
        year_std_weight = years_w_stats[year]['std'] * stdev_w_coeff
        lw_thresh[year] = year_mean_weight - year_std_weight



        ymg = years_g_stats[year]['mean']
        if ymg is None:
            # 2004
            hg_thresh[year] = 0
            continue
        ystdg = years_g_stats[year]['std'] * stdev_gr_coeff
        hg_thresh[year] = ymg + ystdg

    for year in sorted_years:
        topic_growth = topic_info["growths"][year]
        year_mean_growth = years_g_stats[year]['mean']
        year_std_growth = years_g_stats[year]['std'] * stdev_gr_coeff

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
        year_std_weight = years_w_stats[year]['std'] * stdev_w_coeff

        if topic_weight < (year_mean_weight - year_std_weight):
            low_weight_years.append(year)
        else:
            break
    if not low_weight_years:
        return None

    # minimum region length checks
    if num_low_weight is None:
        # has to span the entire year
        if not (high_growth_years + low_weight_years) == sorted_years:
            return None
    else:
        # low weight years has to be enough
        if len(low_weight_years) < num_low_weight:
            return None

    if min_num_high_growth is not None:
        # gotta be the last year sequence
        if consecutive_growth_years:
            if not high_growth_years[-min_num_high_growth:] == sorted_years[:min_num_high_growth]:
                return None
        else:
            # high g years has to be enough
            if len(high_growth_years) < min_num_high_growth:
                return None

    low_weight_years.reverse()
    high_growth_years.reverse()

    visualize_emerging(low_weight_years, high_growth_years, lw_thresh, hg_thresh, topic_info)
    return low_weight_years, high_growth_years

def main():
    # expects a csv with:
    # topicid 	topicname 	year 	weight
    try:
        infile = sys.argv[1]
        exclusion = sys.argv[1]
    except:
        infile = "topic_trends_per_year.csv"
        exclusion = [7, 15, 51, 77, 78, 86, 150, 153, 158, 160, 164, 201, 225, 230, 244, 253, 259, 271, 273, 285, 286, 341, 342, 390, 426, 446, 466, 472, 475, 477, 493]

    print("Reading from ", infile)
    print("Excluding ", len(exclusion), " topics")

    x = pandas.read_csv(infile)
    x = x[x["year"] < 2019]
    if exclusion:
        x = x[~x["topicid"].isin(exclusion)]
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
       topics[topic]["index"] = topic
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

    print("Year-wise statistics means")
    for year in years_weights_stats:
        print(year, years_weights_stats[year]['mean'], years_weights_stats[year]['std'])

    # decide topic types
    emergings, safes, hibernating = [], [], []

    for t in topics:
        topic_info = topics[t]
        topic_info["types"] = []

        num_low_weight = None
        min_num_high_growth = 4
        em_valid_range = [ 2014, 2016, 2017, 2018, 2019]
        # res = check_emerging(topic_info, years_weights_stats, years_growths_stats, num_low_weight,stdev_w_coeff=0)
        # res = check_emerging(topic_info, years_weights_stats, years_growths_stats, num_low_weight, min_num_high_growth=min_num_high_growth, stdev_w_coeff=0.5, consecutive_growth_years=True)
        res = None
        if res is not None:
            low, high = res
            if not em_valid_range or (low[-1] in em_valid_range or high[0] in em_valid_range):
                emergings.append((t, res))
                topic_info["types"].append("E")
        num_safe_years = None
        safebet_coeff = 1
        # res = check_safe_bet(topic_info, years_weights_stats, num_years=num_safe_years, stdev_w_coeff=safebet_coeff)
        res = None
        if res:
            safes.append(t)
            topic_info["types"].append("S")

        min_g_years = None
        policy="full"
        hi_valid_range = list(range(2011, 2013+1))
        coeff = 0.75
        visualize = None
        visualize = "show" # "write"
        visualize = "write"
        g_h_years = check_hibernating(topic_info, years_weights_stats, min_g_years,policy, stdev_w_coeff=coeff, valid_range=hi_valid_range, do_visualize=visualize)
        # g_h_years = None
        if g_h_years is not None:
            hibernating.append((t, g_h_years))
            topic_info["types"].append("H")

    print("Emerging: num low weight years: {}, min high growth rate years: {}, range: {}".format(num_low_weight, min_num_high_growth, em_valid_range))
    for i, e in enumerate(sorted(emergings)):
        topic = e[0]
        last_low = e[1][0][-1]
        print("{}/{} |last low year: {}|: {} {}".format(i+1, len(emergings), last_low, topic, topics[topic]["label"]))
        # if i > 8:
        #     print("...")
        #     break
    print("Safe: num safe years: {}, coeff: {}".format(num_safe_years, safebet_coeff))
    for i, e in enumerate(sorted(safes)):
        print("{}/{} : {} {}".format(i+1, len(safes), e, topics[e]["label"]))
        # if i > 8:
        #     print("...")
        #     break
    print("Hibernating: stdev coeff: {}, range: {}".format(num_safe_years, coeff, hi_valid_range))
    for i, e in enumerate(sorted(hibernating, key=lambda x: x[0])):
        topic, gh_years = e
        giants, hibers = gh_years
        lastgyear, firsthyear = giants[-1], hibers[0]
        print("{}/{} | last giant / first hib years: {}, {} | : {} {}".format(i+1, len(hibernating), lastgyear, firsthyear,
              topic, topics[topic]["label"]))
        # if i > 8:
        #     print("...")
        #     break
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

    return

    x.to_csv("topic_diffs_growths_types.filteredQuality.csv")
#    print("\nmax:\n", x.max())
#    print("\nmin:\n", x.min())
#    print("\nsum:\n", x.sum())

    # write one without the per-year information
    y = x[["topicid","topicname","hibernating","emerging","safebet"]].drop_duplicates()
    y.to_csv("topic_diffs_growths_types.filteredQuality.onlytypes.csv", index=None)


    # plot
    safes = x[x["safebet"] ==1]
    sortedyears = sorted(years_weights.keys())
    all_weights = np.ndarray((0, len(sortedyears)), dtype=np.float32)
    topicnames = []
    for topic in set(list(safes["topicid"])):
        topicnames.append(topics[topic]['label'])
        weights = [0 for _ in sortedyears]
        for year in topics[topic]['years']:
            weights[sortedyears.index(year)] = topics[topic]['years'][year]
        all_weights = np.vstack((all_weights, np.asarray(weights)))
    fix, ax = plt.subplots()
    ax.stackplot(safes["year"].drop_duplicates().sort_values(), *all_weights)
    plt.title("Safe bets")
    plt.legend(topicnames)
    plt.show()


def visualize_hibernating(hyears, gyears, giant_threshold, hib_threshold, topic_info, output="show"):
    plt.figure()
    topic_index = topic_info["index"]
    name = topic_info["label"]
    topic_name = "{}: {}".format(topic_index, name)
    print("Topic ", topic_name)

    weights = topic_info["years"]
    years = sorted(list(giant_threshold.keys()))
    xidx = list(range(len(years)))
    # giant lower limit
    plt.plot(xidx, [giant_threshold[years[i]] for i in xidx],'r')
    # hibernating upper limit
    plt.plot(xidx, [hib_threshold[years[i]] for i in xidx],'g')
    # year mean
    # plt.plot(xidx, [year_means[year] for year in years],'--')
    # year fluctuation
    # plt.plot(xidx, [year_flucts[year] for year in years],'.')
    # print("Fluctuations:", [year_flucts[year] for year in years])
    # print("Means:", [year_means[year] for year in years])
    # leg = ["giant_year_thr","hib_year_thr", "year_mean", "year_fluct"]
    # leg = ["giant_year_thr","hib_year_thr", "year_mean"]
    leg = ["giant_year_thr","hib_year_thr"]


    topic_g, topic_h, topic_neutral = [],[],[]
    for i, year in enumerate(years):
        if year not in weights:
            continue
        weight = weights[year]
        if year in gyears:
            topic_g.append((i, weight))
        elif year in hyears:
            topic_h.append((i, weight))
        else:
            topic_neutral.append((i, weight))
    if topic_g:
        plt.plot(*zip(*topic_g), "r*")
        if "topic giant" not in leg:
            leg.append("topic giant")
    if topic_h:
        plt.plot(*zip(*topic_h), "g*")
        if "topic hib" not in leg:
            leg.append("topic hib")
    if topic_neutral:
        plt.plot(*zip(*topic_neutral), "ko")
        if "topic neutral" not in leg:
            leg.append("topic neutral")

    plt.legend(leg)
    plt.title(topic_name)
    plt.xticks(ticks=range(len(years)), labels=years, rotation='vertical')
    if output == "show":
        plt.show()
    elif output == "write":
        outpath = "hibernating_" + topic_name.replace("/","_") + ".png"
        # print("Writing to", outpath)
        plt.savefig(outpath)
    else:
        print("Visualization", output, "undefined")
        exit(1)

def visualize_safe(topic_info, thresh, years, output="write"):
    plt.figure()
    topic_index = topic_info["index"]
    name = topic_info["label"]
    topic_name = "{}: {}".format(topic_index, name)

    xidx = list(range(len(years)))
    under, over = [], []
    for i, year in enumerate(years):
        w = topic_info["years"][year]
        if w < thresh[i]:
            under.append((i, w))
        else:
            over.append((i, w))


    # thresh
    plt.plot(xidx, thresh,'r')
    # hibernating upper limit
    plt.plot(*zip(*over))
    # hibernating upper limit
    plt.plot(*zip(*under))
    leg = ["safe threshold", "over", "under"]
    # year mean
    # plt.plot(xidx, [year_means[year] for year in years],'--')
    
    plt.legend(leg)
    plt.title(topic_name)
    plt.xticks(ticks=range(len(years)), labels=years, rotation='vertical')
    if output == "show":
        plt.show()
    elif output == "write":
        outpath = "safe_" + topic_name.replace("/","_") + ".png"
        # print("Writing to", outpath)
        plt.savefig(outpath)
    else:
        print("Visualization", output, "undefined")
        exit(1)

def visualize_emerging(low_weight_years, high_growth_years, lw_thresh, hg_thresh, topic_info, output="write"):
    # print(low_weight_years, high_growth_years)
    plt.figure()
    ax_gr = plt.subplot(211)
    ax_we = plt.subplot(212, sharex=ax_gr)

    topic_index = topic_info["index"]
    # if topic_index == 370:
    # print()
    name = topic_info["label"]
    topic_name = "{}: {}".format(topic_index, name)

    years = sorted(lw_thresh.keys())

    xidx = list(range(len(years)))
    ax_we.plot(xidx, [lw_thresh[y] for y in years])
    ax_gr.plot(xidx, [hg_thresh[y] for y in years])

    lw, hw, lg, hg = [], [], [], []
    for i, year in enumerate(years):
        w = topic_info["years"][year]
        if w < lw_thresh[year]:
            lw.append((i, w))
        else:
            hw.append((i, w))

        g = topic_info["growths"][year]
        if year == 2004:
            continue
        if g > hg_thresh[year]:
            hg.append((i, g))
        else:
            lg.append((i, g))

    # hibernating upper limit
    if lw:
        ax_we.plot(*zip(*lw),"*")
    if hw:
        ax_we.plot(*zip(*hw),"*")
    # hibernating upper limit
    if lg:
        ax_gr.plot(*zip(*lg),"*")
    if hg:
        ax_gr.plot(*zip(*hg),"*")

    leg_we = ["low weight threshold","lw topic", "hw topic"]
    leg_gr = ["high growth threshold", "lg topic", "hg topic"]
    # the cutoff
    ax_we.axvline(years.index(low_weight_years[-1]))
    ax_gr.legend(leg_gr)
    ax_we.legend(leg_we)
    
    ax_gr.set_title(topic_name)
    # ax_we.set_title(topic_name)
    plt.xticks(ticks=range(len(years)), labels=years, rotation='vertical')
    # ax_we.xticks(ticks=range(len(years)), labels=years, rotation='vertical')
    if output == "show":
        plt.show()
    elif output == "write":
        outpath = "emerging_" + topic_name.replace("/","_") + ".png"
        # print("Writing to", outpath)
        plt.savefig(outpath)
    else:
        print("Visualization", output, "undefined")
        exit(1)


if __name__ == '__main__':
    main()
