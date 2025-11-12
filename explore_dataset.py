import pandas as pd
import os

# === PATH TO YOUR DOWNLOADED DATASET ===
dataset_path = r"C:\Users\Nicol\.cache\kagglehub\datasets\bwandowando\clash-royale-season-18-dec-0320-dataset\versions\20"

# === Load Card Master List CSV ===
cards_csv = os.path.join(dataset_path, "CardMasterListSeason18_12082020.csv")
cards_df = pd.read_csv(cards_csv)

print("\n==============================")
print(" CARD MASTER LIST (FIRST 5 ROWS) ")
print("==============================")
print(cards_df.head())

# === Load Win Conditions CSV ===
wincons_csv = os.path.join(dataset_path, "Wincons.csv")
wincons_df = pd.read_csv(wincons_csv)

print("\n==============================")
print(" WIN CONDITIONS (FIRST 5 ROWS) ")
print("==============================")
print(wincons_df.head())

# === Load Battles CSV ===
battles_csv = os.path.join(dataset_path, "battles.csv")
battles_df = pd.read_csv(battles_csv)

print("\n==============================")
print(" BATTLES DATA (FIRST 5 ROWS) ")
print("==============================")
print(battles_df.head())

# === Quick info about each dataset ===
print("\n--- DataFrame Sizes ---")
print(f"Card Master List: {cards_df.shape}")
print(f"Wincons: {wincons_df.shape}")
print(f"Battles: {battles_df.shape}")

# === Inspect columns ===
print("\n--- Battle Columns ---")
print(battles_df.columns.tolist())

# === Summary of numerical data (like trophies, elixir, etc.) ===
print("\n--- Battle Stats Summary ---")
print(battles_df.describe())

# === Optional: check missing values ===
print("\n--- Missing Values per Column ---")
print(battles_df.isnull().sum())
