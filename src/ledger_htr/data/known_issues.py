"""Known data-quality issues identified by the Zindi community and confirmed
excludable by the organizers, kept here as a single source of truth rather
than scattered filters.

Per Zindi discussion "Rules clarification: hyperparameter search &
training-data cleaning" (thread 33891): organizer (meganomaly, Zindi staff)
confirmed on 2026-07-16 that clearly corrupted training rows (image/label
mismatches, empty images) may be excluded from training, provided the
original Train.csv is left unchanged and the excluded rows are documented.
This list is the community-compiled + organizer-acknowledged set of 21 IDs
(shared by Alpcan_Cepik in that thread on 2026-07-18); verified present in
our local Train.csv (21/21 found) on 2026-09-04.
"""

CORRUPTED_TRAIN_IDS: frozenset[str] = frozenset(
    {
        "79tMUVyfIdy3GzkG",
        "rh8o7bdCGOIBFPwH",
        "F8DYDDp2AvW9Dytw",
        "mfQxfOmeRmwBh0g8",
        "EcxuqKeZl7OQexfB",
        "PO7QQLWIFOT65BTz",
        "yNyf3Tp0zc7DFj5F",
        "JU7lRwk3jKkus24Z",
        "R6iYPb7MHFiHtXH6",
        "VmrEALeZiP1Y6nF9",
        "t0UrASljcgzvBAnO",
        "KH5g92Q3DA5Bo6xi",
        "N78M3v6GKmUzF7sk",
        "0CCrVKAom8EK53jj",
        "3ZxOeKcOr5wUYyk0",
        "PJbM7Q1SrblrSWt6",
        "WwlTCykxjP3c4kfo",
        "baTY3OlGskirWgFc",
        "u3b4JNo5bqpfE7Js",
        "MfT9S5oghk9ywNSC",
        "8H2ITJSWZhAD6eh0",
    }
)
