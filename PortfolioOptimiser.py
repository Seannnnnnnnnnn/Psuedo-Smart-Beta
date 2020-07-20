import numpy as np
import pandas as pd
import datetime as dt
import csv
from matplotlib import pyplot as plt
from yahooquery import Ticker


# first two functions format data. If using on different system remember to update file paths
# TODO: Tidy up compile_data()
# TODO: Fit to theoretical distribution
# TODO: Decompose the main() into separate funcs


def collect_data():
    # fill tickers with the security code from IOZ ETF data
    start_date, end_date = dt.datetime(2018, 1, 1), dt.datetime(2020, 7, 17)

    securities = []
    with open('IOZ_holdings.csv') as etf_data:
        reader = csv.reader(etf_data)
        for row in reader:
            securities.append(row[0] + '.AX')

    for ticker in securities:
        # determine which securities have price history & which others we will have to treat separately
        try:
            data = Ticker(ticker).history(start=start_date, end=end_date)
            dates = data.index.values # grabs associated date
            data.insert(0,'Date',dates)
            data.to_csv('/Users/MichaelConlon/Desktop/Data/{}.csv'.format(ticker), index=False, header=True)

            print('Completed: {}'.format(ticker))

        except Exception as error:
            print('Error encountered with {}'.format(ticker))
            print('Exception: {}'.format(error))
    print('Data gathered')


def compile_data():
    # grab securities code:
    securities = []
    with open('IOZ_holdings.csv') as etf_data:
        reader = csv.reader(etf_data)
        for row in reader:
            securities.append(row[0] + '.AX')

    # combine into data frame
    main_df = pd.DataFrame()

    for count, ticker in enumerate(securities):
        try:
            # some securities contain different headers hence nested try
            # TODO: This could be cleaner - remove nested try-except block
            try:
                df = pd.read_csv('/Users/MichaelConlon/Desktop/Data/{}.csv'.format(ticker))
                df.set_index('Date', inplace=True)
                df.rename(columns = {'adjclose': ticker}, inplace=True)
                df.drop(['open','high','close','volume','low','dividends'],1,
                        inplace=True) # only keep adjusted closed price

                main_df = main_df.join(df, how='outer')


            except Exception as error:
                print('Inital error occured with: {} Exception: {}'.format(ticker, error))
                df = pd.read_csv('/Users/MichaelConlon/Desktop/Data/{}.csv'.format(ticker))
                df.set_index('Date', inplace=True)
                df.rename(columns={'adjclose': ticker}, inplace=True)
                df.drop(['open', 'high', 'close', 'volume', 'low'], 1,
                        inplace=True)  # only keep adjusted closed price

                main_df = main_df.join(df, how='outer')


            print('Completed for: {}'.format(ticker))
        except Exception as error:
            print('Error encountered with : {}'.format(ticker))
            print('Exception: {}'.format(error))

    return main_df


def marchenko_pastur_pdf(x, Q, sigma=1):
    # Marcenko Pastur distribution
    y = 1 / Q
    b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)  # Largest eigenvalue
    a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)  # Smallest eigenvalue
    return (1 / (2 * np.pi * sigma * sigma * x * y)) * np.sqrt((b - x) * (x - a)) * (0 if (x > b or x < a) else 1)


def compare_eigenvalue_distribution(random_matrix, Q, sigma=1, set_autoscale=True, show_top=True):
    # compares eigenvalues of correlation_matrix to the theoretical distribution of the Marcenko Pastur PDF

    assert random_matrix.shape[0] == random_matrix.shape[1]  # random matrix must me square

    e, _ = np.linalg.eig(random_matrix)  # Correlation matrix is Hermitian, so this is faster than other variants of eig


    x_max = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if not show_top:
        # Clear top eigenvalue from plot
        e = e[e <= x_max + 1]
    ax.hist(e, normed=True, bins=50)  # Histogram the eigenvalues
    ax.set_autoscale_on(set_autoscale)

    # Plot the theoretical density
    f = np.vectorize(lambda x: marchenko_pastur_pdf(x, Q, sigma=sigma))

    x_min = .0001 if np.power(sigma * (1 - np.sqrt(1 / Q)), 2) < .0001 else np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
    x_max = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)

    x = np.linspace(x_min, x_max, 5000)
    ax.plot(x, f(x), linewidth=4, color='r', label='Marcenko Pastur pdf')
    plt.title('Eigenvalues Against Theoretical PDF')
    plt.xlabel('λ')
    plt.legend()
    plt.show()


def get_cumulative_returns_over_time(sample, weights):
    # returns cumulative returns of a sample portfolio of given weights
    weights[weights <= 0 ] = 0
    weights = weights / weights.sum()
    return (((1+sample).cumprod(axis=0))-1).dot(weights)


def main():
    # runs the script


    ####################################################################
    #### Meta-Data
    ####################################################################


    prices = compile_data()
    print('\n\nTotal observations (Pre-Parse)', prices.shape)
    returns = prices.pct_change()
    returns = returns.iloc[1:, :]  # Remove first row of NA's generated by pct_changes()
    returns.dropna(axis = 1, thresh=len(returns.index)/2, inplace=True)  # Drop stocks with over half the data missing
    returns.dropna(axis = 0, thresh=len(returns.columns), inplace=True)  # Drop days without data for all stocks
    training_period = 100
    in_sample = returns.iloc[:(returns.shape[0]-training_period), :].copy()
    tickers = in_sample.columns  # Remove tickers that were dropped

    print('Total observations (Post-Parse): ', returns.shape)  # displays matrix dimensions


    # apply Log Transformation to in_sample data:
    log_in_sample = in_sample.apply(lambda x: np.log(x + 1))
    log_in_sample.dropna(0, inplace=True)  # Drop those NA
    log_in_sample.dropna(1, inplace=True)

    # We will need the standard deviations later:
    variances = np.diag(log_in_sample.cov().values)
    standard_deviations = np.sqrt(variances)

    # compare empirical eigenvalues to Marcenko Pastur
    T, N = returns.shape  # Pandas does the reverse of what I wrote in the first section
    Q = T / N
    correlation_matrix = log_in_sample.interpolate().corr()
    compare_eigenvalue_distribution(correlation_matrix, Q, set_autoscale=True)

    # Let's see the eigenvalues larger than the largest theoretical eigenvalue
    sigma = 1  # The variance for all of the standardized log returns is 1
    max_theoretical_eval = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)

    D, S = np.linalg.eigh(correlation_matrix)
    # Filter the eigenvalues out
    D[D <= max_theoretical_eval] = 0

    # Reconstruct the matrix
    temp = np.dot(S, np.dot(np.diag(D), np.transpose(S)))

    # Set the diagonal entries to 0
    np.fill_diagonal(temp, 1)
    filtered_matrix = temp

    f = plt.figure()
    ax = plt.subplot(121)
    ax.imshow(correlation_matrix)
    plt.title("Original")
    ax = plt.subplot(122)
    plt.title("Filtered")
    a = ax.imshow(filtered_matrix)
    cbar = f.colorbar(a, ticks=[-1, 0, 1])
    plt.show()


    ####################################################################
    #### Portfolio Optimisation
    ####################################################################


    # Reconstruct the filtered covariance matrix:
    covariance_matrix = in_sample.cov()
    inv_cov_mat = np.linalg.pinv(covariance_matrix)

    # Construct minimum variance weights
    ones = np.ones(len(inv_cov_mat))
    inv_dot_ones = np.dot(inv_cov_mat, ones)
    min_var_weights = inv_dot_ones / np.dot(inv_dot_ones, ones)

    plt.figure(figsize=(16, 4))

    ax = plt.subplot(121)
    min_var_portfolio = pd.DataFrame(data=min_var_weights,
                                     columns=['Investment Weight'],
                                     index=tickers)
    min_var_portfolio.plot(kind='bar', ax=ax)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.title('Minimum Variance')

    # Reconstruct the filtered covariance matrix from the standard deviations and the filtered correlation matrix
    filtered_cov = np.dot(np.diag(standard_deviations),
                          np.dot(filtered_matrix, np.diag(standard_deviations)))

    filt_inv_cov = np.linalg.pinv(filtered_cov)

    # Construct minimum variance weights
    ones = np.ones(len(filt_inv_cov))
    inv_dot_ones = np.dot(filt_inv_cov, ones)
    filt_min_var_weights = inv_dot_ones / np.dot(inv_dot_ones, ones)
    ax = plt.subplot(122)
    filt_min_var_portfolio = pd.DataFrame(data=filt_min_var_weights,
                                          columns=['Investment Weight'],
                                          index=tickers)
    filt_min_var_portfolio.plot(kind='bar', ax=ax)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.title('Filtered Minimum Variance')

    print(filt_min_var_portfolio.head())

    in_sample_ind = np.arange(0, (returns.shape[0] - training_period + 1))
    out_sample_ind = np.arange((returns.shape[0] - training_period), returns.shape[0])

    cumulative_returns = get_cumulative_returns_over_time(returns, min_var_portfolio).values
    cumulative_returns_filt = get_cumulative_returns_over_time(returns, filt_min_var_portfolio).values

    f = plt.figure(figsize=(16, 4))

    ax = plt.subplot(131)
    ax.plot(cumulative_returns[in_sample_ind], 'black')
    ax.plot(out_sample_ind, cumulative_returns[out_sample_ind], 'r')
    plt.title("Minimum Variance Portfolio")

    ax = plt.subplot(132)
    ax.plot(cumulative_returns_filt[in_sample_ind], 'black')
    ax.plot(out_sample_ind, cumulative_returns_filt[out_sample_ind], 'r')
    plt.title("Filtered Minimum Variance Portfolio")


    ax = plt.subplot(133)
    ax.plot(cumulative_returns, 'black')
    ax.plot(cumulative_returns_filt, 'b')
    plt.title("Filtered (Blue) vs. Normal (Black)")

    plt.show()




main()
