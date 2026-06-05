import os
import torch

from multitask_inference import predict_multitask

examples = [
    # obvious jokes
    "I used to play piano by ear, but now I use my hands.",
    "My cat looked at the expensive food I bought and decided starvation was more dignified.",
    "I told my computer I needed a break, and now it won't stop sending me vacation ads.",
    "Parallel lines have so much in common. It's a shame they'll never meet.",

    # obvious non-jokes
    "The database backup completed successfully.",
    "The meeting will start tomorrow at 10 AM.",
    "Please reset your password before logging in.",
    "MariaDB replication is currently healthy.",

    # mixed / borderline
    "I spent two hours fixing a bug and accidentally deleted the fix.",
    "My salary arrived and immediately developed separation anxiety from my bank account.",
    "The printer is working as expected.",
    "I bought a gym membership and gained confidence instead of muscles.",

    # offensive / controversial examples
    "What sound does a gay magician make when he disappears? Poof.",
    "Why do all Asians think they're a celebrity in a crowded place? Because everyone wants a selfie.",
]

print("=" * 80)
print("MULTI-TASK BERTWEET TEST")
print("=" * 80)

for text in examples:
    try:
        result = predict_multitask(text)

        print()
        print("-" * 80)
        print(text)
        print("-" * 80)

        print(f"Label:                 {result['label']}")
        print(f"Confidence:            {result['confidence']:.4f}")

        print(f"Not humorous prob:     {result['not_humorous_probability']:.4f}")
        print(f"Humorous prob:         {result['humorous_probability']:.4f}")

        print(f"Humor rating:          {result['humor_rating_0_to_5']:.2f}/5")
        print(f"Offense rating:        {result['offense_rating_0_to_5']:.2f}/5")

        print(f"Controversial:         {result['controversial']}")
        print(f"Controversy prob:      {result['controversy_probability']:.4f}")

    except Exception as e:
        print()
        print(text)
        print("ERROR:", e)

print()
print("=" * 80)
print("DONE")
print("=" * 80)