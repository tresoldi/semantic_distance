#!/usr/bin/env python3
# encoding: utf-8

# Import Python standard libraries
import argparse
import datetime
from itertools import chain, combinations, islice, tee
from operator import itemgetter
import os.path

# Import external libraries
import networkx as nx
import scipy


def _pairwise(iterable):
    """
    Internal function for sequential pairwise iteration.

    The function follows the recipe in Python's itertools documentation.

    "s -> (s[0], s[1]), (s[1], s[2]), (s[2], s[3]) ...
    """

    item_a, item_b = tee(iterable)
    next(item_a, None)

    return zip(item_a, item_b)


# todo: CACHE, exchangin memory for speed (no `graph` query) -- but consider
# birectionality
def comp_weight(path, graph):
    """
    Compute the cumulative weight associated with a path in a graph.
    """

    return sum([
        graph.edges[(edge[0], edge[1])]['weight']
        for edge in _pairwise(path)
        ])


# Note to self: really wish a functional lazy evaluation here...
def output_distances(graph, args):
    """
    Output the distances and paths for all potential pairs in the graph.
    """

    buffer = []

    # Open output handler and write headers (generated accounting for the
    # requested number of k best paths); we also cache the length of the
    # headers, so we don compute it repeatedly later
    handler = open(os.path.join(args.output, "distances.tsv"), "a")
    headers = ['id', 'comb_idx', 'concept_a', 'concept_b', 'distance'] + \
        list(chain.from_iterable([
            ["path-%i" % i, "steps-%i" % i, "weight-%i" % i] for i in range(args.k)]
        ))
    if args.suboptimal:
        headers += ["path-so", "steps-so", "weight-so"]
    headers_len = len(headers)
    if args.start == 0:
        buffer.append("\t".join(headers))

    # Collect data for all possible combinations, operating on the sorted
    # list of concept glosses; we need to use a counter, instead of the
    # index from the enumeration, as we will skip over combinations with
    # no paths; we also cache the number of total combinations
    ncr = scipy.special.comb(len(graph.nodes), 2)
    row_count = 1
    for comb_idx, comb in enumerate(combinations(sorted(graph.nodes), 2)):
        # Skip what was already done
        if comb_idx < args.start:
            continue
    
        # See if we should write/report
        if comb_idx % 100 == 0:
            print("[%s] Writing until combination #%i (row count %i)..." %
                (datetime.datetime.now(), comb_idx, row_count))
            for row in buffer:
                handler.write(row)
                handler.write("\n")
            handler.flush()
            buffer = []
  
            print("[%s] Processing combination #%i/%i..." %
                (datetime.datetime.now(), comb_idx, ncr))

        # Collect args.paths shortest paths for the combination, skipping
        # over if there is no path for the current combination. This will
        # collect a higher number of paths, so we can look for the weight of
        # the best path that does not include the intermediate steps of the
        # single best path. Note that we will compute all weights and sort,
        # as the implementation of Yen's algorithm is not returning the
        # best paths in order.
        # TODO: what if the single best is a direct one?
        try:
            k_paths = list(islice(
                nx.shortest_simple_paths(graph, comb[0], comb[1], weight='weight'),
                args.search))
        except:
            # no path
            continue

        # Compute the cumulative weight associated with each path --
        # unfortunately, `nx.shortest_simple_paths` does not return it
        k_weights = [comp_weight(path, graph) for path in k_paths]

        # Build a sorted list of (path, weights) elements
        # TODO see if it can be faster, probably removing list, perhaps
        # a list comprehension for the zip
        paths = list(zip(k_paths, k_weights))
        paths = sorted(paths, key=itemgetter(1))

        # Get the sub-optimal best path without the intermediate steps of
        # the best global path; if no exclude path is found, we will use the
        # score from the worst one we collected
        if not args.suboptimal:
            paths = paths[:args.k]
        else:
            excludes = set(chain.from_iterable([path[0][1:-1] for path in paths[:args.k]]))
            exclude_paths = [
                path for path in paths
                if not any([concept in path[0] for concept in excludes])
            ]

            if exclude_paths:
                paths = paths[:args.k] + [exclude_paths[0]]
            else:
                paths = paths[:args.k] + [paths[-1]]

        # For easier manipulation, extract list of concepts and weights and
        # proceed building the output
        concept_paths, weights = zip(*paths)

        # Turn paths and weights into a strings and collect the number of steps
        steps = [str(len(path)-2) for path in concept_paths]
        path_strs = ["/".join(path) for path in concept_paths]
        weights_strs = ["%0.2f" % weight for weight in weights]

        # Build buffer and write
        # TODO can cache len(weights) -- but what if suboptimal? an if?
        computed_data = chain.from_iterable(zip(path_strs, steps, weights_strs))
        buf = [
            str(row_count),
            str(comb_idx),
            comb[0],
            comb[1],
            "%0.2f" % (sum(weights)/len(weights)), # distance
        ] + list(computed_data)

        # Add empty items to the list if necessary
        buf += [""] * (headers_len - len(buf))

        # Write to handler and update counter
        buffer.append("\t".join(buf))
        row_count += 1
        
        if row_count == 300:
            break

    # Close handler and return
    handler.close()


def main(args):
    """
    Main function, reading data and generating output.
    """
    
    # Input graph
    graph = nx.read_gml("output/graph.gml")    
    
    # Output the distance for all possible pairs
    output_distances(graph, args)


if __name__ == "__main__":
    # Define the parser for when called from the command-line
    parser = argparse.ArgumentParser(description="Compute semantic shift distances.")
    parser.add_argument(
        "start",
        type=int,
        help="Number of the first combination, for resuming.")
    parser.add_argument(
        "--f_dexp",
        type=float,
        help="Denominator exponent for family count correction (default: 3.0)",
        default=1.0)
    parser.add_argument(
        "--l_dexp",
        type=float,
        help="Denominator exponent for language count correction (default: 2.0)",
        default=1.2)
    parser.add_argument(
        "--w_dexp",
        type=float,
        help="Denominator exponent for word count correction (default: 1.0)",
        default=1.4)
    parser.add_argument(
        "--cluster_exp",
        type=float,
        help="Exponent for same cluster correction (default: 0.9)",
        default=0.9)
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the data directory (default: 'data')",
        default="data")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output directory (default: 'data')",
        default="output")
    parser.add_argument(
        "-k",
        type=int,
        help="Maximum number of best paths to collect for each pair (default: 3)",
        default=3)
    parser.add_argument(
        "--search",
        type=int,
        help="Multiplier for the search space of best suboptimal path (default: 5)",
        default=5)
    parser.add_argument(
        '--suboptimal',
        action='store_true',
        help="Whether to search for suboptimal paths (expansive, default: False)")
    ARGS = parser.parse_args()

    main(ARGS)
