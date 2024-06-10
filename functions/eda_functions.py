def single_summary_plot(df, tic_freq):

    statistics = df.describe()

    plt.figure(figsize=(6, 4))
    for index, row in statistics.iterrows():
        plt.plot(statistics.columns, row, label=index)

    plt.xlabel('Column')
    plt.ylabel('Value')
    plt.title('Summary Statistics')
    plt.legend(loc = 'lower right')
    plt.xticks(statistics.columns[::tic_freq])
    plt.show()

def summary_plot(dfs, dfs_names):

    # subplot dimensions
    num_dfs = len(dfs)
    num_rows = num_dfs // 2 + num_dfs % 2

    # plot
    plt.figure(figsize=(15, 5 * num_rows))
    for i, df in enumerate(dfs):

        statistics = df.describe()

        plt.subplot(num_rows, 2, i+1)
        for index, row in statistics.iterrows():
            plt.plot(statistics.columns, row, label=index)

        plt.xlabel('Column')
        plt.ylabel('Value')
        plt.title(f'{dfs_names[i]} Summary')
        plt.xticks(statistics.columns[::2])
        plt.legend(loc = 'center right')

    plt.tight_layout()
    plt.show()
