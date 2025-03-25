import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16
})

party_colors = {
    'CDU/CSU': 'black',
    'SPD': 'red',
    'GRUENE': 'green',
    'AfD': 'blue',
    'FDP': 'yellow',
    'DIE LINKE': 'purple',
    'parteilos': 'grey',
    'Die PARTEI': 'darkred',
    'LKR': 'darkblue',
    'all': 'darkgrey',
    'Unknown': 'lightgrey',
}

file_path = "new_speeches_output.csv"
df = pd.read_csv(file_path)

df['Party'] = df['Party'].replace({'CDU': 'CDU/CSU', 'CSU': 'CDU/CSU', 'Univ Kyiv': 'CDU/CSU', 'UnivKyiv':'CDU/CSU', 'Erlangen':'CDU/CSU', 'BÜNDNIS 90/D': 'GRUENE', 'BÜNDNISSES 90/DIE GRÜNEN': 'GRUENE','BÜNDNIS 90/DIE GRÜNEN': 'GRUENE', 'BÜNDIS 90/DIE GRÜNEN': 'GRUENE', 'LINKEN': 'DIE LINKE', 'LINKE': 'DIE LINKE'})
df['Interjector Party'] = df['Interjector Party'].replace({'CDU': 'CDU/CSU', 'CSU': 'CDU/CSU', 'Univ Kyiv': 'CDU/CSU', 'UnivKyiv':'CDU/CSU', 'Erlangen':'CDU/CSU', 'BÜNDNIS 90/D': 'GRUENE', 'BÜNDNISSES 90/DIE GRÜNEN': 'GRUENE','BÜNDNIS 90/DIE GRÜNEN': 'GRUENE', 'BÜNDIS 90/DIE GRÜNEN': 'GRUENE', 'LINKEN': 'DIE LINKE', 'LINKE': 'DIE LINKE'})

interjections_df = df[df['Interjection'] == True]


nonverbal_interjections_df = df[df['Nonverbal interjection'] == True]

verbal_interjections_df = df[df['Verbal interjection'] == True]

def plot_general(df, color_scheme):
    # Get only speech, no interjection
    last_5_dates = df['Date'].dropna().unique()
    last_5_dates.sort()
    last_5_dates = last_5_dates[-5:]

    df = df[df['Date'].isin(last_5_dates)]

    speech_df = df[df['Interjection'] == False]

    paragraphs_per_date_party = speech_df.groupby(['Date', 'Party']).size().unstack(fill_value=0)
    print(paragraphs_per_date_party)
    # get only interjection
    interjections_df = df[df['Interjection'] == True]

    nonverbal_interjections_df = df[df['Nonverbal interjection'] == True]

    verbal_interjections_df = df[df['Verbal interjection'] == True]

    # Count the total, verbal, and nonverbal interjections
    total_interjections = len(interjections_df)
    nonverbal_count = len(nonverbal_interjections_df)
    verbal_count = len(verbal_interjections_df)

    # Calculate the ratios
    nonverbal_ratio = nonverbal_count / total_interjections if total_interjections > 0 else 0
    verbal_ratio = verbal_count / total_interjections if total_interjections > 0 else 0

    print(f"Total Interjections: {total_interjections}")
    print(f"Nonverbal Interjections: {nonverbal_count} ({nonverbal_ratio:.2%})")
    print(f"Verbal Interjections: {verbal_count} ({verbal_ratio:.2%})")

    ### Verbal plots (received interjections)
    verbal_counts_per_date_party = verbal_interjections_df.groupby(['Date', 'Party']).size().unstack(fill_value=0)
    print(verbal_counts_per_date_party)

    normalized_verbal_counts = verbal_counts_per_date_party.div(paragraphs_per_date_party, axis=0).fillna(0)


    ax = normalized_verbal_counts.plot(
        kind='bar',
        figsize=(15, 8),
        stacked=False,
        color=[party_colors.get(party, 'lightgrey') for party in normalized_verbal_counts.columns]  # Match colors to parties
    )

    plt.title('Normalized Verbal Interjections received by Party and Date (per Non-Interjection Paragraph per Date)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Interjections (per Non-Interjection Paragraph)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Party', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.show()


    ### Nonverbal plots (received interjections)
    nonverbal_counts_per_date_party = nonverbal_interjections_df.groupby(['Date', 'Party']).size().unstack(fill_value=0)

    normalized_nonverbal_counts = nonverbal_counts_per_date_party.div(paragraphs_per_date_party, axis=0).fillna(0)

    ax = normalized_nonverbal_counts.plot(
        kind='bar',
        figsize=(15, 8),
        stacked=False,
        color=[party_colors.get(party, 'lightgrey') for party in normalized_nonverbal_counts.columns]  # Match colors to parties
    )

    plt.title('Normalized Nonverbal Interjections received by Party and Date (per Non-Interjection Paragraph per Date)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Interjections (per Non-Interjection Paragraph)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Party', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    ### Verbal plots (given interjections)
    verbal_counts_per_date_party = verbal_interjections_df.groupby(['Date', 'Interjector Party']).size().unstack(fill_value=0)

    normalized_verbal_counts = verbal_counts_per_date_party.div(paragraphs_per_date_party, axis=0).fillna(0)

    ax = normalized_verbal_counts.plot(
        kind='bar',
        figsize=(15, 8),
        stacked=False,
        color=[party_colors.get(party, 'lightgrey') for party in normalized_verbal_counts.columns]  # Match colors to parties
    )

    plt.title('Normalized Verbal Interjections given by Party and Date (per Non-Interjection Paragraph per Date)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Interjections (per Non-Interjection Paragraph)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Party', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.show()


    ### Nonverbal plots (given interjections)
    nonverbal_counts_per_date_party = nonverbal_interjections_df.groupby(['Date', 'Interjector Party']).size().unstack(fill_value=0)

    normalized_nonverbal_counts = nonverbal_counts_per_date_party.div(paragraphs_per_date_party, axis=0).fillna(0)

    ax = normalized_nonverbal_counts.plot(
        kind='bar',
        figsize=(15, 8),
        stacked=False,
        color=[party_colors.get(party, 'lightgrey') for party in normalized_nonverbal_counts.columns]  # Match colors to parties
    )

    plt.title('Normalized Nonverbal Interjections given by Party and Date (per Non-Interjection Paragraph per Date)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Interjections (per Non-Interjection Paragraph)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Party', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.show()


def plot_interjections_received(df, interjection_type, party_colors):
    """
    Plot interjections (verbal or nonverbal) received by parties.

    Parameters:
        df (DataFrame): Data containing interjection details.
        interjection_type (str): Type of interjection ('Verbal' or 'Nonverbal').
        party_colors (dict): Dictionary mapping parties to colors.
    """

    if interjection_type not in ['Verbal', 'Nonverbal']:
        raise ValueError("interjection_type must be 'Verbal' or 'Nonverbal'")

    interjection_col = f"{interjection_type} interjection"
    last_5_dates = df['Date'].dropna().unique()
    last_5_dates.sort()
    last_5_dates = last_5_dates[-5:]

    df = df[df['Date'].isin(last_5_dates)]

    filtered_df = df[df[interjection_col] == True]

    counts_per_party = (
        filtered_df.groupby(['Date', 'Party', 'Interjector Party'])
        .size()
        .unstack(fill_value=0)
    )

    # List of unique parties
    parties = counts_per_party.index.get_level_values('Party').unique()

    # Create a grid of subplots
    num_parties = len(parties)
    ncols = 3
    nrows = (num_parties + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 12), sharey=True)
    axes = axes.flatten()

    # Plot data for each party
    for i, party in enumerate(parties):
        # Filter data for the current party
        party_data = counts_per_party.loc[(slice(None), party), :]
        party_data = party_data.reset_index(level='Party', drop=True)

        ax = axes[i]
        party_data.plot(
            kind='bar',
            stacked=False,
            ax=ax,
            color=[party_colors.get(interjector_party, 'grey') for interjector_party in party_data.columns]
        )

        ax.set_title(f"{interjection_type} Interjections Received by {party}")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Num of {interjection_type} Interjections")
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Interjection Party', loc='upper left', fontsize=8)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def plot_interjections_given(df, interjection_type, party_colors):
    """
    Plot interjections (verbal or nonverbal) received by parties.

    Parameters:
        df (DataFrame): Data containing interjection details.
        interjection_type (str): Type of interjection ('Verbal' or 'Nonverbal').
        party_colors (dict): Dictionary mapping parties to colors.
    """
    if interjection_type not in ['Verbal', 'Nonverbal']:
        raise ValueError("interjection_type must be 'Verbal' or 'Nonverbal'")

    interjection_col = f"{interjection_type} interjection"

    last_5_dates = df['Date'].dropna().unique()
    last_5_dates.sort()
    last_5_dates = last_5_dates[-5:]

    df = df[df['Date'].isin(last_5_dates)]

    filtered_df = df[df[interjection_col] == True]

    counts_per_party = (
        filtered_df.groupby(['Date', 'Interjector Party', 'Party', ])
        .size()
        .unstack(fill_value=0)
    )

    parties = counts_per_party.index.get_level_values('Interjector Party').unique()

    num_parties = len(parties)
    ncols = 3
    nrows = (num_parties + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 12), sharey=True)
    axes = axes.flatten()

    for i, party in enumerate(parties):
        party_data = counts_per_party.loc[(slice(None), party), :]
        party_data = party_data.reset_index(level='Interjector Party', drop=True)

        ax = axes[i]
        party_data.plot(
            kind='bar',
            stacked=False,
            ax=ax,
            color=[party_colors.get(party, 'grey') for party in party_data.columns]
        )

        ax.set_title(f"{interjection_type} Interjections Given by {party}")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Num of {interjection_type} Interjections")
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Party', loc='upper left', fontsize=8)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



### VERBAL TOP INTERJECTOR

def plot_interjector(df, color):
    last_5_dates = df['Date'].dropna().unique()
    last_5_dates.sort()
    last_5_dates = last_5_dates[-5:]

    df_last_5 = df[df['Date'].isin(last_5_dates)]
    interjections_last_5_df = df_last_5[df_last_5['Interjection'] == True]

    verbal_interjections_5_df = interjections_last_5_df[interjections_last_5_df['Verbal interjection'] == True]

    df_grouped = verbal_interjections_5_df.groupby(['Date', 'Interjector', 'Interjector Party']).size().reset_index(name='Count')
    df_grouped_sorted = df_grouped.sort_values(['Date', 'Interjector Party', 'Count'], ascending=[True, True, False])

    #  For each Date and Party, get the top 3 speakers
    top_speakers = df_grouped_sorted.groupby(['Date', 'Interjector Party']).head(3)

    unique_parties = top_speakers['Interjector Party'].unique()
    ncols = min(len(unique_parties), 3)
    nrows = (len(unique_parties) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    for i, party in enumerate(unique_parties):
        party_data = top_speakers[top_speakers['Interjector Party'] == party]

        party_pivot = party_data.pivot_table(index='Date', columns='Interjector', values='Count', aggfunc='sum', fill_value=0)

        ax = axes[i]
        party_pivot.plot(kind='bar', stacked=True, ax=ax)

        ax.set_title(f'Top 3 Interjectors from {party}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Interjections')
        ax.legend(title='Interjector', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    plt.show()

    df_grouped = verbal_interjections_5_df.groupby(['Date', 'Interjector', 'Interjector Party']).size().reset_index(name='Count')

    date_totals = df_grouped.groupby('Date')['Count'].sum().reset_index(name='Total_Interjections')

    df_grouped = df_grouped.merge(date_totals, on='Date')

    df_grouped['Normalized_Count'] = df_grouped['Count'] / df_grouped['Total_Interjections']

    df_grouped_sorted = df_grouped.sort_values(['Date', 'Normalized_Count'], ascending=[True, False])
    filtered_df = df_grouped_sorted[
        ~df_grouped_sorted['Interjector'].isin(['Unknown', 'all'])
    ]

    top_5_per_date = filtered_df.groupby('Date').head(5)

    dates = top_5_per_date['Date'].unique()
    for date in dates:
        date_data = top_5_per_date[top_5_per_date['Date'] == date]

    # Map party colors to the data
        colors = date_data['Interjector Party'].map(color)
        colors = colors.fillna('gray')

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(date_data['Interjector'], date_data['Normalized_Count'], color=colors)
        plt.gca().invert_yaxis()
        plt.title(f'Top 5 Interjectors on {date}', fontsize=14)
        plt.xlabel('Normalized Proportion of Interjections', fontsize=12)
        plt.ylabel('Interjector', fontsize=12)
        plt.tight_layout()
        plt.show()


    # third plot
    df_grouped = verbal_interjections_df.groupby(['Interjector', 'Interjector Party']).size().reset_index(name='Count')

    total_interjections = df_grouped['Count'].sum()
    df_grouped['Normalized_Count'] = df_grouped['Count'] / total_interjections
    df_grouped_sorted = df_grouped.sort_values('Normalized_Count', ascending=False)

    filtered_df = df_grouped_sorted[
        ~df_grouped_sorted['Interjector'].isin(['Unknown', 'all'])
    ]

    # Select the top 10 interjectors overall
    top_10_overall = filtered_df.head(10)


    colors = top_10_overall['Interjector Party'].map(party_colors)

    plt.figure(figsize=(10, 6))
    plt.barh(top_10_overall['Interjector'], top_10_overall['Normalized_Count'], color=colors)
    plt.gca().invert_yaxis()  # Invert y-axis to show the highest at the top
    plt.title('Top 10 Interjectors (verbal) Overall', fontsize=14)
    plt.xlabel('Normalized Proportion of Interjections', fontsize=12)
    plt.ylabel('Interjector', fontsize=12)


    plt.tight_layout()
    plt.show()


def plot_type_interjection(df, color):
    nonverbal_interjections_df = df[df['Nonverbal interjection'] == True]

    df_grouped = nonverbal_interjections_df.groupby(
        ['Date', 'Interjection type', 'Interjector Party']
    ).size().reset_index(name='Count')

    party_pivot = df_grouped.pivot_table(
        index='Interjector Party',
        columns='Interjection type',
        values='Count',
        aggfunc='sum',
        fill_value=0
    )

    plt.figure(figsize=(12, 8))

    party_pivot.plot(kind='bar', stacked=True, figsize=(12, 8))

    plt.title('Distribution of Nonverbal Interjection Types by Party', fontsize=16)
    plt.xlabel('Interjector Party', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(title='Speaker party', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_type_verbal_interjection(df, color):
    verbal_interjections_df = df[df['Verbal interjection'] == True]

    df_grouped = verbal_interjections_df.groupby(
        ['Date', 'Interjection type', 'Interjector Party']
    ).size().reset_index(name='Count')

    party_pivot = df_grouped.pivot_table(
        index='Interjector Party',
        columns='Interjection type',
        values='Count',
        aggfunc='sum',
        fill_value=0
    )

    plt.figure(figsize=(12, 8))

    party_pivot.plot(kind='bar', stacked=True, figsize=(12, 8))

    plt.title('Distribution of Verbal Interjection Types by Party', fontsize=16)
    plt.xlabel('Interjector Party', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(title='Speaker party', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_interjection_distribution_given(df, color):
    """
    Plot interjection distributions for each interjector party, showing speaker parties on the x-axis
    and interjection types as stacked bars (normalized by the overall number of that interjection type).

    Args:
        df (pd.DataFrame): DataFrame containing interjection data with columns 'Interjector Party', 'Speaker Party', and 'Interjection type'.
        color (dict): Dictionary mapping interjection types to colors.
    """
    nonverbal_interjections_df = df[df['Nonverbal interjection'] == True]

    grouped = nonverbal_interjections_df.groupby([
        'Interjector Party', 'Party', 'Interjection type'
    ]).size().reset_index(name='Count')

    interjector_parties = grouped['Interjector Party'].unique()

    for interjector_party in interjector_parties:
        # Filter data for the current interjector party
        party_data = grouped[grouped['Interjector Party'] == interjector_party]

        pivot = party_data.pivot_table(
            index='Interjection type',
            columns='Party',
            values='Count',
            aggfunc='sum',
            fill_value=0
        )

        pivot_normalized = pivot.div(pivot.sum(axis=1), axis=0)

        pivot_normalized.plot(
            kind='bar',
            stacked=True,
            figsize=(10, 6),
            color=color
        )

        plt.title(f'Normalized Nonverbal Interjections Given by {interjector_party}', fontsize=16)
        plt.xlabel('Interjection Type', fontsize=14)
        plt.ylabel('Proportion of Interjections', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.legend(title='Speaker Party', fontsize=12)
        plt.tight_layout()

        plt.show()



def plot_interjection_distribution_received(df, color):
    """
    Plot interjection distributions for each interjector party, showing speaker parties on the x-axis
    and interjection types as stacked bars.

    Args:
        df (pd.DataFrame): DataFrame containing interjection data with columns 'Interjector Party', 'Speaker Party', and 'Interjection type'.
        color (dict): Dictionary mapping interjection types to colors.
    """
    nonverbal_interjections_df = df[df['Nonverbal interjection'] == True]

    grouped = nonverbal_interjections_df.groupby([
        'Interjector Party', 'Party', 'Interjection type'
    ]).size().reset_index(name='Count')

    all_parties = grouped['Party'].unique()

    for party in all_parties:
        party_data = grouped[grouped['Party'] == party]

        pivot = party_data.pivot_table(
            index='Interjection type',
            columns='Interjector Party',
            values='Count',
            aggfunc='sum',
            fill_value=0
        )

        pivot_normalized = pivot.div(pivot.sum(axis=1), axis=0)

        pivot_normalized.plot(#
            kind='bar',
            stacked=True,
            figsize=(10, 6),
            color=color
        )

        plt.title(f'Nonverbal Interjections received by {party}', fontsize=16)
        plt.xlabel('Count of Interjections', fontsize=14)
        plt.ylabel('Interjection Type', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.legend(title='Interjector Party', fontsize=12)
        plt.tight_layout()
        plt.show()



def plot_party_interactions_heatmap(df):
    """
    Plot a heatmap showing interactions between interjector parties and speaker parties.

    Args:
        df (pd.DataFrame): DataFrame containing interjection data with columns 'Interjector Party' and 'Speaker Party'.
    """
    grouped = df.groupby(['Interjector Party', 'Party']).size().reset_index(name='Count')

    heatmap_data = grouped.pivot_table(
        index='Interjector Party',
        columns='Party',
        values='Count',
        aggfunc='sum',
        fill_value=0
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='Purples', fmt="d", linewidths=0.5)
    plt.title('Heatmap of Interjections by Interjector and Speaker Party', fontsize=16)
    plt.xlabel('Speaker Party', fontsize=14)
    plt.ylabel('Interjector Party', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_party_nv_interactions_heatmap(df):
    """
    Plot a heatmap showing interactions between interjector parties and speaker parties.

    Args:
        df (pd.DataFrame): DataFrame containing interjection data with columns 'Interjector Party' and 'Speaker Party'.
    """
    nonverbal_interjections_df = df[df['Nonverbal interjection'] == True]
    grouped = nonverbal_interjections_df.groupby(['Interjector Party', 'Party']).size().reset_index(name='Count')

    heatmap_data = grouped.pivot_table(
        index='Interjector Party',
        columns='Party',
        values='Count',
        aggfunc='sum',
        fill_value=0
    )

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='BuPu', fmt="d", linewidths=0.5)
    plt.title('Heatmap of Nonverbal Interjections by Interjector and Speaker Party', fontsize=16)
    plt.xlabel('Speaker Party', fontsize=14)
    plt.ylabel('Interjector Party', fontsize=14)
    plt.tight_layout()
    plt.show()



def plot_party_v_interactions_heatmap(df):
    """
    Plot a heatmap showing interactions between interjector parties and speaker parties.

    Args:
        df (pd.DataFrame): DataFrame containing interjection data with columns 'Interjector Party' and 'Speaker Party'.
    """
    verbal_interjections_df = df[df['Verbal interjection'] == True]
    grouped = verbal_interjections_df.groupby(['Interjector Party', 'Party']).size().reset_index(name='Count')

    heatmap_data = grouped.pivot_table(
        index='Interjector Party',
        columns='Party',
        values='Count',
        aggfunc='sum',
        fill_value=0
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='Greens', fmt="d", linewidths=0.5)
    plt.title('Heatmap of Verbal Interjections by Interjector and Speaker Party', fontsize=16)
    plt.xlabel('Speaker Party', fontsize=14)
    plt.ylabel('Interjector Party', fontsize=14)
    plt.tight_layout()
    plt.show()


def speaker_heatmap(df):
    verbal_interjections_df = df[df['Verbal interjection'] == True]

    verbal_interjections_df = verbal_interjections_df[
        (verbal_interjections_df['Speaker'] != 'Unknown') &
        (verbal_interjections_df['Interjector'] != 'Unknown')
        ]

    verbal_interjections_df['Speaker_with_Party'] = (
            verbal_interjections_df['Speaker'] + " (" + verbal_interjections_df['Party'] + ")"
    )
    verbal_interjections_df['Interjector_with_Party'] = (
            verbal_interjections_df['Interjector'] + " (" + verbal_interjections_df['Interjector Party'] + ")"
    )

    interjector_counts = verbal_interjections_df.groupby('Interjector_with_Party')['Speaker'].count().nlargest(3)
    top_interjectors = interjector_counts.index  # Get the names of the top 10 interjectors

    filtered_df = verbal_interjections_df[
        verbal_interjections_df['Interjector_with_Party'].isin(top_interjectors)
    ]

    interaction_counts = filtered_df.groupby(
        ['Speaker_with_Party', 'Interjector_with_Party']
    ).size().reset_index(name='Count')


    interaction_matrix = interaction_counts.pivot(
        index='Speaker_with_Party',
        columns='Interjector_with_Party',
        values='Count'
    ).fillna(0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(interaction_matrix, annot=True, fmt='.0f', cmap='Blues', linewidths=0.5)


    plt.title('Number of Interjections: Interjectors to Speakers')
    plt.xlabel('Interjector')
    plt.ylabel('Speaker')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def find_role_switches(df):
    verbal_interjections_df = df[df['Verbal interjection'] == True]

    # Exclude "Unknown"
    verbal_interjections_df = verbal_interjections_df[
        (verbal_interjections_df['Speaker'] != 'Unknown') &
        (verbal_interjections_df['Interjector'] != 'Unknown')
        ]

    # Create sets of (Speaker, Interjector) and reversed (Interjector, Speaker)
    interactions = set(
        zip(verbal_interjections_df['Speaker'], verbal_interjections_df['Interjector'])
    )
    reversed_interactions = set(
        zip(verbal_interjections_df['Interjector'], verbal_interjections_df['Speaker'])
    )

    # Find matches where roles switch
    role_switches = interactions.intersection(reversed_interactions)

    role_switch_details = verbal_interjections_df[
        verbal_interjections_df.apply(
            lambda row: (row['Speaker'], row['Interjector']) in role_switches or
                        (row['Interjector'], row['Speaker']) in role_switches,
            axis=1
        )
    ]

    interaction_counts = role_switch_details.groupby(
        ['Speaker', 'Interjector']
    ).size().reset_index(name='Count')

    interaction_matrix = interaction_counts.pivot(
        index='Speaker',
        columns='Interjector',
        values='Count'
    ).fillna(0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(interaction_matrix, annot=True, fmt='.0f', cmap='Blues', linewidths=0.5)

    plt.title('Number of Interjections: Interjectors to Speakers')
    plt.xlabel('Interjector')
    plt.ylabel('Speaker')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



# Call the function to plot the heatmap
plot_general(df, party_colors)
plot_interjections_received(df=df, interjection_type='Verbal', party_colors=party_colors)
plot_interjections_received(df=df, interjection_type='Nonverbal', party_colors=party_colors)

plot_interjections_given(df=df, interjection_type='Verbal', party_colors=party_colors)
plot_interjections_given(df=df, interjection_type='Nonverbal', party_colors=party_colors)

plot_interjector(df, party_colors)
plot_type_interjection(df, party_colors)
plot_type_verbal_interjection(df, party_colors)
plot_interjection_distribution_given(df, party_colors)
plot_interjection_distribution_received(df, party_colors)


plot_party_interactions_heatmap(df)
plot_party_nv_interactions_heatmap(df)
plot_party_v_interactions_heatmap(df)


find_role_switches(df)
