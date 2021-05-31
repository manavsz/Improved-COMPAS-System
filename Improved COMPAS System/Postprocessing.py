from utils import apply_threshold, get_num_predicted_positives, get_total_accuracy, get_positive_predictive_value, \
    get_num_correct, get_true_positive_rate

#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: Accuracy
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02.
    Chooses the best solution of those that satisfy this constraint based on chosen
    secondary optimization criteria.
"""


def enforce_demographic_parity(categorical_results, epsilon):

    target_probability = 0
    target_probability_list = {}
    accurate_target_list = {}
    max_accuracy = float('-inf')
    demographic_parity_data = {'African-American': [], 'Caucasian': [], 'Hispanic': [], 'Other': []}
    thresholds = {}

    # Find candidate thresholds
    while target_probability <= 1:
        target_probability_list[target_probability] = []
        count = 0

        # African-American
        threshold = 0
        flag = False
        while threshold <= 1:
            threshdolded_values = apply_threshold(categorical_results['African-American'], threshold)
            ppra = get_num_predicted_positives(threshdolded_values) / len(categorical_results['African-American'])

            if abs(ppra - target_probability) <= epsilon:
                target_probability_list[target_probability].append({'African-American': (threshold, threshdolded_values)})
                count += 1
                flag = True
                break
            threshold += 0.01
        if not flag:
            target_probability += 0.01
            continue

        # Caucasian
        threshold = 0
        flag = False
        while threshold <= 1:
            threshdolded_values = apply_threshold(categorical_results['Caucasian'], threshold)
            pprc = get_num_predicted_positives(threshdolded_values) / len(categorical_results['Caucasian'])

            if abs(pprc - target_probability) <= epsilon:
                target_probability_list[target_probability].append({'Caucasian': (threshold, threshdolded_values)})
                count += 1
                flag = True
                break
            threshold += 0.01
        if not flag:
            target_probability += 0.01
            continue

        # Hispanic
        threshold = 0
        flag = False
        while threshold <= 1:
            threshdolded_values = apply_threshold(categorical_results['Hispanic'], threshold)
            pprh = get_num_predicted_positives(threshdolded_values) / len(categorical_results['Hispanic'])

            if abs(pprh - target_probability) <= epsilon:
                target_probability_list[target_probability].append({'Hispanic': (threshold, threshdolded_values)})
                count += 1
                flag = True
                break
            threshold += 0.01
        if not flag:
            target_probability += 0.01
            continue

        # Other
        threshold = 0
        flag = False
        while threshold <= 1:
            threshdolded_values = apply_threshold(categorical_results['Other'], threshold)
            ppro = get_num_predicted_positives(threshdolded_values) / len(categorical_results['Other'])

            if abs(ppro - target_probability) <= epsilon:
                target_probability_list[target_probability].append({'Other': (threshold, threshdolded_values)})
                count += 1
                flag = True
                break
            threshold += 0.01
        if count == 4:
            accurate_target_list[target_probability] = target_probability_list[target_probability]
        target_probability += 0.01

    # Take the candidate thresholds and return one with most accuracy
    keys = accurate_target_list.keys()
    for key in keys:
        threshold_instance = accurate_target_list[key]

        temp_demographic_parity_data = {'African-American': threshold_instance[0]['African-American'][1],
                                   'Caucasian': threshold_instance[1]['Caucasian'][1],
                                   'Hispanic': threshold_instance[2]['Hispanic'][1],
                                   'Other': threshold_instance[3]['Other'][1]}

        acc = get_total_accuracy(temp_demographic_parity_data)

        if acc > max_accuracy:
            max_accuracy = acc
            demographic_parity_data = temp_demographic_parity_data
            thresholds = {'African-American': threshold_instance[0]['African-American'][0], 'Caucasian': threshold_instance[1]['Caucasian'][0], 'Hispanic': threshold_instance[2]['Hispanic'][0], 'Other': threshold_instance[3]['Other'][0]}

    return demographic_parity_data, thresholds


#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon,
    and chooses best solution according to chosen secondary optimization criteria. For the Naive
    Bayes Classifier and SVM you should be able to find a non-'trivial solution with epsilon=0.01
"""


def enforce_equal_opportunity(categorical_results, epsilon):

    target_probability = 0
    target_probability_list = {}
    accurate_target_list = {}
    max_accuracy = float('-inf')
    equal_opportunity_data = {'African-American': [], 'Caucasian': [], 'Hispanic': [], 'Other': []}
    thresholds = {}

    # Find candidate thresholds
    while target_probability <= 1:
        target_probability_list[target_probability] = []
        count = 0

        # African-American
        threshold = 0
        flag = False
        while threshold <= 1:
            threshdolded_values = apply_threshold(categorical_results['African-American'], threshold)
            ppv = get_true_positive_rate(threshdolded_values)

            if abs(ppv - target_probability) <= epsilon:
                target_probability_list[target_probability].append({'African-American': (threshold, threshdolded_values)})
                count += 1
                flag = True
                break
            threshold += 0.01
        if not flag:
            target_probability += 0.01
            continue

        # Caucasian
        threshold = 0
        flag = False
        while threshold <= 1:
            threshdolded_values = apply_threshold(categorical_results['Caucasian'], threshold)
            ppv = get_true_positive_rate(threshdolded_values)

            if abs(ppv - target_probability) <= epsilon:
                target_probability_list[target_probability].append({'Caucasian': (threshold, threshdolded_values)})
                count += 1
                flag = True
                break
            threshold += 0.01
        if not flag:
            target_probability += 0.01
            continue

        # Hispanic
        threshold = 0
        flag = False
        while threshold <= 1:
            threshdolded_values = apply_threshold(categorical_results['Hispanic'], threshold)
            ppv = get_true_positive_rate(threshdolded_values)

            if abs(ppv - target_probability) <= epsilon:
                target_probability_list[target_probability].append({'Hispanic': (threshold, threshdolded_values)})
                count += 1
                flag = True
                break
            threshold += 0.01
        if not flag:
            target_probability += 0.01
            continue

        # Other
        threshold = 0
        flag = False
        while threshold <= 1:
            threshdolded_values = apply_threshold(categorical_results['Other'], threshold)
            ppv = get_true_positive_rate(threshdolded_values)

            if abs(ppv - target_probability) <= epsilon:
                target_probability_list[target_probability].append({'Other': (threshold, threshdolded_values)})
                count += 1
                flag = True
                break
            threshold += 0.01
        if count == 4:
            accurate_target_list[target_probability] = target_probability_list[target_probability]

        target_probability += 0.01

    # Take the candidate thresholds and return one with most accuracy
    keys = accurate_target_list.keys()
    for key in keys:
        threshold_instance = accurate_target_list[key]

        temp_parity_data = {'African-American': threshold_instance[0]['African-American'][1],
                            'Caucasian': threshold_instance[1]['Caucasian'][1],
                            'Hispanic': threshold_instance[2]['Hispanic'][1],
                            'Other': threshold_instance[3]['Other'][1]}

        acc = get_total_accuracy(temp_parity_data)

        if acc > max_accuracy:
            max_accuracy = acc
            equal_opportunity_data = temp_parity_data
            thresholds = {'African-American': threshold_instance[0]['African-American'][0], 'Caucasian': threshold_instance[1]['Caucasian'][0],'Hispanic': threshold_instance[2]['Hispanic'][0],'Other': threshold_instance[3]['Other'][0]}

    return equal_opportunity_data, thresholds


#######################################################################################################################
"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""


def enforce_maximum_profit(categorical_results):

    mp_data = {'African-American': [], 'Caucasian': [], 'Hispanic': [], 'Other': []}
    thresholds = {'African-American': [], 'Caucasian': [], 'Hispanic': [], 'Other': []}

    # African American
    threshold = 0
    max_accuracy = float('-inf')
    while threshold <= 1:
        threshold_african_american = apply_threshold(categorical_results['African-American'], threshold)
        acc = get_num_correct(threshold_african_american)

        if acc > max_accuracy:
            max_accuracy = acc
            mp_data['African-American'] = threshold_african_american
            thresholds['African-American'] = threshold

        threshold += .01

    # Caucasian
    threshold = 0
    max_accuracy = float('-inf')
    while threshold <= 1:
        threshold_african_american = apply_threshold(categorical_results['Caucasian'], threshold)
        acc = get_num_correct(threshold_african_american)

        if acc > max_accuracy:
            max_accuracy = acc
            mp_data['Caucasian'] = threshold_african_american
            thresholds['Caucasian'] = threshold

        threshold += .01

    # Hispanic
    threshold = 0
    max_accuracy = float('-inf')
    while threshold <= 1:
        threshold_african_american = apply_threshold(categorical_results['Hispanic'], threshold)
        acc = get_num_correct(threshold_african_american)

        if acc > max_accuracy:
            max_accuracy = acc
            mp_data['Hispanic'] = threshold_african_american
            thresholds['Hispanic'] = threshold

        threshold += .01

    # Other
    threshold = 0
    max_accuracy = float('-inf')
    while threshold <= 1:
        threshold_african_american = apply_threshold(categorical_results['Other'], threshold)
        acc = get_num_correct(threshold_african_american)

        if acc > max_accuracy:
            max_accuracy = acc
            mp_data['Other'] = threshold_african_american
            thresholds['Other'] = threshold

        threshold += .01

    return mp_data, thresholds


#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""


def enforce_predictive_parity(categorical_results, epsilon):

    target_probability = 0
    target_probability_list = {}
    accurate_target_list = {}
    max_accuracy = float('-inf')
    predictive_parity_data = {'African-American': [], 'Caucasian': [], 'Hispanic': [], 'Other': []}
    thresholds = {}

    # Find candidate thresholds
    while target_probability <= 1:
        target_probability_list[target_probability] = []
        count = 0

        # African-American
        threshold = 0
        flag = False
        while threshold <= 1:
            threshdolded_values = apply_threshold(categorical_results['African-American'], threshold)
            ppv = get_positive_predictive_value(threshdolded_values)

            if abs(ppv - target_probability) <= epsilon:
                target_probability_list[target_probability].append({'African-American': (threshold, threshdolded_values)})
                count += 1
                flag = True
                break
            threshold += 0.01
        if not flag:
            target_probability += 0.01
            continue

        # Caucasian
        threshold = 0
        flag = False
        while threshold <= 1:
            threshdolded_values = apply_threshold(categorical_results['Caucasian'], threshold)
            ppv = get_positive_predictive_value(threshdolded_values)

            if abs(ppv - target_probability) <= epsilon:
                target_probability_list[target_probability].append({'Caucasian': (threshold, threshdolded_values)})
                count += 1
                flag = True
                break
            threshold += 0.01
        if not flag:
            target_probability += 0.01
            continue

        # Hispanic
        threshold = 0
        flag = False
        while threshold <= 1:
            threshdolded_values = apply_threshold(categorical_results['Hispanic'], threshold)
            ppv = get_positive_predictive_value(threshdolded_values)

            if abs(ppv - target_probability) <= epsilon:
                target_probability_list[target_probability].append({'Hispanic': (threshold, threshdolded_values)})
                count += 1
                flag = True
                break
            threshold += 0.01
        if not flag:
            target_probability += 0.01
            continue

        # Other
        threshold = 0
        flag = False
        while threshold <= 1:
            threshdolded_values = apply_threshold(categorical_results['Other'], threshold)
            ppv = get_positive_predictive_value(threshdolded_values)

            if abs(ppv - target_probability) <= epsilon:
                target_probability_list[target_probability].append({'Other': (threshold, threshdolded_values)})
                count += 1
                flag = True
                break
            threshold += 0.01
        if count == 4:
            accurate_target_list[target_probability] = target_probability_list[target_probability]
        target_probability += 0.01

    # Take the candidate thresholds and return one with most accuracy
    keys = accurate_target_list.keys()
    for key in keys:
        threshold_instance = accurate_target_list[key]

        temp_parity_data = {'African-American': threshold_instance[0]['African-American'][1],
                            'Caucasian': threshold_instance[1]['Caucasian'][1],
                            'Hispanic': threshold_instance[2]['Hispanic'][1],
                            'Other': threshold_instance[3]['Other'][1]}

        acc = get_total_accuracy(temp_parity_data)

        if acc > max_accuracy:
            max_accuracy = acc
            predictive_parity_data = temp_parity_data
            thresholds = {'African-American': threshold_instance[0]['African-American'][0], 'Caucasian': threshold_instance[1]['Caucasian'][0],'Hispanic': threshold_instance[2]['Hispanic'][0],'Other': threshold_instance[3]['Other'][0]}

    return predictive_parity_data, thresholds


#######################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to
    chosen secondary optimization criteria
"""


def enforce_single_threshold(categorical_results):

    single_threshold_data = {}
    thresholds = {}

    # iterate through 0-1 thresholds with increments of 0.01, set max accuracy threshold to thresholds
    threshold = 0
    max_accuracy = float('-inf')
    while threshold <= 1:

        threshold_african_american = apply_threshold(categorical_results['African-American'], threshold)
        threshold_caucasian = apply_threshold(categorical_results['Caucasian'], threshold)
        threshold_hispanic = apply_threshold(categorical_results['Hispanic'], threshold)
        threshold_other = apply_threshold(categorical_results['Other'], threshold)

        temp_data = {'African-American': threshold_african_american, 'Caucasian': threshold_caucasian, 'Hispanic': threshold_hispanic, 'Other': threshold_other}
        acc = get_total_accuracy(temp_data)

        if acc > max_accuracy:
            max_accuracy = acc
            single_threshold_data = temp_data
            thresholds = {'African-American': threshold, 'Caucasian': threshold, 'Hispanic': threshold, 'Other': threshold}

        threshold += 0.01

    return single_threshold_data, thresholds
