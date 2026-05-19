import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('report_outputs', exist_ok=True)

print("🎨 MEMBUAT MASTER GRAFIK LENGKAP (4 PANEL) UNTUK BAB 4...")

try:
    df = pd.read_csv('report_outputs/model_comparison_metrics.csv')

    # Bikin Grid 2x2 (2 Baris, 2 Kolom)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    sns.set_theme(style="whitegrid")

    # Warna: Merah buat LSTM, Hijau buat Random Forest
    custom_palette = {'LSTM': '#D9383A', 'Random Forest': '#45B39D'}

    def add_labels(ax, is_integer=False):
        """Fungsi pembantu buat nambahin angka di atas bar"""
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                text = format(int(height), 'd') if is_integer else format(height, '.2f')
                ax.annotate(text, (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 8), 
                            textcoords='offset points', weight='bold', fontsize=10)

    # ─── PANEL 1 (Kiri Atas): PRECISION ───
    sns.barplot(data=df, x='City', y='Precision_Heatwave', hue='Model', ax=axes[0, 0], palette=custom_palette)
    axes[0, 0].set_title('1. Precision (Akurasi Prediksi Heatwave)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].set_xlabel('')
    add_labels(axes[0, 0])

    # ─── PANEL 2 (Kanan Atas): RECALL ───
    sns.barplot(data=df, x='City', y='Recall_Heatwave', hue='Model', ax=axes[0, 1], palette=custom_palette)
    axes[0, 1].set_title('2. Recall (Keberhasilan Mendeteksi Heatwave Asli)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].set_xlabel('')
    add_labels(axes[0, 1])

    # ─── PANEL 3 (Kiri Bawah): F1-SCORE ───
    sns.barplot(data=df, x='City', y='F1_Score_Heatwave', hue='Model', ax=axes[1, 0], palette=custom_palette)
    axes[1, 0].set_title('3. F1-Score (Keseimbangan Precision & Recall)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].set_xlabel('Kota', fontsize=12)
    add_labels(axes[1, 0])

    # ─── PANEL 4 (Kanan Bawah): FALSE POSITIVE ───
    sns.barplot(data=df, x='City', y='FP', hue='Model', ax=axes[1, 1], palette=custom_palette)
    axes[1, 1].set_title('4. False Positives (Jumlah Peringatan Dini Palsu)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Kota', fontsize=12)
    axes[1, 1].set_ylabel('Jumlah Hari', fontsize=12)
    add_labels(axes[1, 1], is_integer=True)

    # Rapihkan layout biar judul gak tabrakan
    plt.tight_layout(pad=4.0)
    plt.savefig('report_outputs/master_thesis_chart_full.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ BERHASIL! Grafik 4-Panel Master telah disimpan di: report_outputs/master_thesis_chart_full.png")

except FileNotFoundError:
    print("❌ ERROR: File model_comparison_metrics.csv tidak ditemukan!")