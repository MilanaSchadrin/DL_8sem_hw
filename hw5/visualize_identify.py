import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import wandb

def get_image_paths(folder_path):
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_paths = set()
    for extension in image_extensions:
        for path in glob(os.path.join(folder_path, '**', extension), recursive=True):
            canonical_path = os.path.realpath(path)
            image_paths.add(canonical_path)
    return sorted(image_paths)

def load_ground_truth(gt_file):
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(';')
    persons = header[1:]
    gt = {}
    for line in lines[1:]:
        parts = line.strip().split(';')
        filename = parts[0]
        labels = []
        for x in parts[1:]:
            if x!='':
                labels.append(int(x))
        gt[filename] = {person: label for person, label in zip(persons, labels)}
    return gt, persons

def evaluate_predictions(gt_file, reference_person, matches, threshold, output_file='predictions_report.txt'):
    gt, persons = load_ground_truth(gt_file)
    if reference_person not in persons:
        raise ValueError(f"Эталонный человек {reference_person} не найден в файле меток")
    results = []
    y_true = []
    y_pred = []
    matched_files = [os.path.basename(p) for p, _ in matches]
    with open(output_file, 'w', encoding='utf-8') as f:
        for filename in gt.keys():
            base_name = os.path.basename(filename)
            true_label = gt[filename][reference_person]
            y_true.append(true_label)
            if base_name in matched_files:
                pred_dist = next(dist for p, dist in matches if os.path.basename(p) == base_name)
                pred_label = 1 if pred_dist <= threshold else 0
            else:
                pred_label = 0
            y_pred.append(pred_label)
            results.append({
                'filename': base_name,
                'true_label': true_label,
                'pred_label': pred_label,
                'correct': true_label == pred_label
            })

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        wandb.log({
            "evaluation_accuracy": accuracy,
            "evaluation_precision": precision,
            "evaluation_recall": recall,
            "evaluation_f1": f1,
            "threshold": threshold
        })

        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")

    return results, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }


def visualize_comparison(reference_path, matches, non_matches, threshold, ref_im, save_to_file=True):
    ref_img = Image.open(reference_path)
    ref_name = os.path.basename(reference_path)
    max_per_row = 5
    row_height = 4
    title_font = 14
    label_font = 11
    n_match_rows = (len(matches) + max_per_row - 1) // max_per_row
    n_non_match_rows = (len(non_matches) + max_per_row - 1) // max_per_row
    n_rows = 1 + n_match_rows + n_non_match_rows
    fig = plt.figure(figsize=(18, row_height * n_rows))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    ax_ref = plt.subplot(n_rows, 1, 1)
    plt.imshow(ref_img)
    plt.title(f"Эталонное изображение: {ref_name}\n",
              fontsize=title_font, pad=20)
    plt.axis('off')

    def plot_image_row(images, title, row_pos, is_match=True):
        ax_title = plt.subplot(n_rows, max_per_row, (row_pos - 1) * max_per_row + 1)
        plt.text(0.5, 0.5, title,
                 fontsize=title_font,
                 ha='center', va='center',
                 color='green' if is_match else 'red')
        plt.axis('off')
        for i in range(min(max_per_row - 1, len(images))):
            ax = plt.subplot(n_rows, max_per_row, (row_pos - 1) * max_per_row + i + 2)
            path, dist = images[i]
            try:
                img = Image.open(path)
                plt.imshow(img)
                label_color = 'green' if is_match else 'red'
                label = f"{ref_im if is_match else 'Not ' + ref_im}\n{os.path.basename(path)}\n"
                plt.title(label, fontsize=label_font, color=label_color, pad=10)
                for spine in ax.spines.values():
                    spine.set_edgecolor(label_color)
                    spine.set_linewidth(2)
            except Exception as e:
                plt.text(0.5, 0.5, f"Ошибка:\n{str(e)}",
                         ha='center', va='center', color='red')
            plt.axis('off')
        for i in range(len(images), max_per_row - 1):
            ax = plt.subplot(n_rows, max_per_row, (row_pos - 1) * max_per_row + i + 2)
            plt.axis('off')

    for i in range(n_match_rows):
        start_idx = i * (max_per_row - 1)
        end_idx = start_idx + (max_per_row - 1)
        plot_image_row(matches[start_idx:end_idx],
                       f"Совпавшие ",
                       i + 2, True)

    for i in range(n_non_match_rows):
        start_idx = i * (max_per_row - 1)
        end_idx = start_idx + (max_per_row - 1)
        plot_image_row(non_matches[start_idx:end_idx],
                       f"Несовпавшие ",
                       i + 2 + n_match_rows, False)

    plt.tight_layout(pad=8.0)
    if save_to_file:
        os.makedirs("comparison_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_results/comparison_{ref_im}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        #print(f"Результат сохранен в файл: {filename}")
    plt.show()