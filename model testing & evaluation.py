# Complete Model Testing and Evaluation Script

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime
import pandas as pd

# Define paths - Update these to your test set paths
TEST_IMG_PATH = "D:/Image Segmentation/preprocessed_images/test/images/"
TEST_MSK_PATH = "D:/Image Segmentation/Adult tooth segmentation dataset/data_split/test/masks/"
MODEL_PATH = "best_model_fast.keras"


# Custom metrics (same as training)
def jaccard_index(y_true, y_pred):
    """Calculates the Jaccard index (IoU)"""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return intersection / (total + tf.keras.backend.epsilon())


def dice_coefficient(y_true, y_pred):
    """Calculates the Dice coefficient"""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + tf.keras.backend.epsilon())


# Additional evaluation metrics
def pixel_accuracy(y_true, y_pred):
    """Calculate pixel-wise accuracy"""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])
    return tf.reduce_mean(tf.cast(tf.equal(y_true_f, y_pred_f), tf.float32))


def sensitivity_recall(y_true, y_pred):
    """Calculate sensitivity/recall (True Positive Rate)"""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])

    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    actual_positives = tf.reduce_sum(y_true_f)

    return true_positives / (actual_positives + tf.keras.backend.epsilon())


def specificity(y_true, y_pred):
    """Calculate specificity (True Negative Rate)"""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])

    true_negatives = tf.reduce_sum((1 - y_true_f) * (1 - y_pred_f))
    actual_negatives = tf.reduce_sum(1 - y_true_f)

    return true_negatives / (actual_negatives + tf.keras.backend.epsilon())


def precision(y_true, y_pred):
    """Calculate precision (Positive Predictive Value)"""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])

    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    predicted_positives = tf.reduce_sum(y_pred_f)

    return true_positives / (predicted_positives + tf.keras.backend.epsilon())


class ModelTester:
    def __init__(self, model_path, test_img_path, test_mask_path):
        self.model_path = model_path
        self.test_img_path = test_img_path
        self.test_mask_path = test_mask_path
        self.model = None
        self.test_files = []
        self.results = []

    def load_model(self):
        """Load the trained model"""
        print("Loading trained model...")
        try:
            # Load model with custom objects
            custom_objects = {
                'jaccard_index': jaccard_index,
                'dice_coefficient': dice_coefficient
            }
            self.model = keras.models.load_model(self.model_path, custom_objects=custom_objects)
            print("Model loaded successfully!")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_test_files(self):
        """Get list of test files"""
        if os.path.exists(self.test_img_path):
            self.test_files = [f for f in os.listdir(self.test_img_path)
                               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f"Found {len(self.test_files)} test images")
            return len(self.test_files) > 0
        else:
            print(f"Test image path does not exist: {self.test_img_path}")
            return False

    def preprocess_image(self, img_path, target_size=(256, 256)):
        """Preprocess a single image"""
        img = tf.keras.preprocessing.image.load_img(
            img_path, color_mode='grayscale', target_size=target_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        return img_array

    def preprocess_mask(self, mask_path, target_size=(256, 256)):
        """Preprocess a single mask"""
        mask = tf.keras.preprocessing.image.load_img(
            mask_path, color_mode='grayscale', target_size=target_size
        )
        mask_array = tf.keras.preprocessing.image.img_to_array(mask)
        mask_array = mask_array / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)
        return mask_array

    def predict_single_image(self, img_file):
        """Predict on a single image and return all processing steps"""
        try:
            # Load and preprocess image
            img_path = os.path.join(self.test_img_path, img_file)
            # Load original image using PIL
            original_img = Image.open(img_path).convert('L')  # Convert to grayscale
            original_img_array = np.array(original_img)
            preprocessed_img = self.preprocess_image(img_path)

            # Load ground truth mask
            mask_file = img_file.replace('_preprocessed.jpg', '.bmp')
            mask_path = os.path.join(self.test_mask_path, mask_file)

            ground_truth = None
            if os.path.exists(mask_path):
                ground_truth = self.preprocess_mask(mask_path)

            # Make prediction
            input_batch = np.expand_dims(preprocessed_img, axis=0)
            prediction = self.model.predict(input_batch, verbose=0)[0]

            # Apply threshold to get binary mask
            binary_prediction = (prediction > 0.5).astype(np.float32)

            # Calculate metrics if ground truth is available
            metrics = {}
            if ground_truth is not None:
                # Convert to tensors for metric calculation
                gt_tensor = tf.constant(ground_truth)
                pred_tensor = tf.constant(prediction)

                metrics = {
                    'dice': float(dice_coefficient(gt_tensor, pred_tensor).numpy()),
                    'jaccard': float(jaccard_index(gt_tensor, pred_tensor).numpy()),
                    'pixel_accuracy': float(pixel_accuracy(gt_tensor, pred_tensor).numpy()),
                    'sensitivity': float(sensitivity_recall(gt_tensor, pred_tensor).numpy()),
                    'specificity': float(specificity(gt_tensor, pred_tensor).numpy()),
                    'precision': float(precision(gt_tensor, pred_tensor).numpy())
                }

            return {
                'filename': img_file,
                'original_image': original_img_array,
                'preprocessed_image': preprocessed_img,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'binary_prediction': binary_prediction,
                'metrics': metrics
            }

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            return None

    def visualize_single_prediction(self, result, save_path=None):
        """Visualize the prediction pipeline for a single image"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Prediction Pipeline: {result['filename']}", fontsize=16, fontweight='bold')

        # Original Image
        axes[0, 0].imshow(result['original_image'], cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Preprocessed Image
        axes[0, 1].imshow(result['preprocessed_image'].squeeze(), cmap='gray')
        axes[0, 1].set_title('Preprocessed Image')
        axes[0, 1].axis('off')

        # Ground Truth (if available)
        if result['ground_truth'] is not None:
            axes[0, 2].imshow(result['ground_truth'].squeeze(), cmap='gray')
            axes[0, 2].set_title('Ground Truth Mask')
        else:
            axes[0, 2].text(0.5, 0.5, 'No Ground Truth\nAvailable',
                            ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Ground Truth Mask')
        axes[0, 2].axis('off')

        # Raw Prediction (probability map)
        im1 = axes[1, 0].imshow(result['prediction'].squeeze(), cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title('Prediction Probability')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # Binary Prediction
        axes[1, 1].imshow(result['binary_prediction'].squeeze(), cmap='gray')
        axes[1, 1].set_title('Binary Prediction (>0.5)')
        axes[1, 1].axis('off')

        # Overlay visualization
        if result['ground_truth'] is not None:
            # Create overlay: GT in green, Prediction in red, overlap in yellow
            overlay = np.zeros((*result['ground_truth'].squeeze().shape, 3))
            gt = result['ground_truth'].squeeze()
            pred = result['binary_prediction'].squeeze()

            # True Positives (overlap) - Yellow
            overlap = gt * pred
            overlay[overlap > 0] = [1, 1, 0]

            # False Positives (pred but not gt) - Red
            fp = pred * (1 - gt)
            overlay[fp > 0] = [1, 0, 0]

            # False Negatives (gt but not pred) - Green
            fn = gt * (1 - pred)
            overlay[fn > 0] = [0, 1, 0]

            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title('Overlay (GT: Green, Pred: Red, Overlap: Yellow)')
        else:
            # Create prediction overlay on original image using numpy operations
            original_normalized = result['original_image'] / 255.0 if result['original_image'].max() > 1 else result[
                'original_image']
            overlay_img = np.stack([original_normalized, original_normalized, original_normalized], axis=-1)
            pred_mask = result['binary_prediction'].squeeze()

            # Resize prediction mask to match original image if needed
            if overlay_img.shape[:2] != pred_mask.shape:
                from PIL import Image
                pred_mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8))
                pred_mask_pil = pred_mask_pil.resize(overlay_img.shape[1::-1], Image.NEAREST)
                pred_mask = np.array(pred_mask_pil) / 255.0

            # Apply red color to prediction areas
            overlay_img[pred_mask > 0.5] = [1, 0, 0]  # Red for predictions
            axes[1, 2].imshow(overlay_img)
            axes[1, 2].set_title('Prediction Overlay')
        axes[1, 2].axis('off')

        # Add metrics text
        if result['metrics']:
            metrics_text = '\n'.join([f"{k.capitalize()}: {v:.4f}" for k, v in result['metrics'].items()])
            fig.text(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

    def test_on_sample_images(self, num_samples=5):
        """Test on a sample of images with detailed visualization"""
        print(f"\n=== Testing on {num_samples} Sample Images ===")

        # Select random sample
        sample_files = np.random.choice(self.test_files,
                                        min(num_samples, len(self.test_files)),
                                        replace=False)

        sample_results = []

        for i, img_file in enumerate(sample_files):
            print(f"\nProcessing {i + 1}/{len(sample_files)}: {img_file}")

            result = self.predict_single_image(img_file)
            if result is not None:
                sample_results.append(result)

                # Print metrics for this image
                if result['metrics']:
                    print("Metrics:")
                    for metric, value in result['metrics'].items():
                        print(f"  {metric.capitalize()}: {value:.4f}")

                # Visualize this prediction
                self.visualize_single_prediction(result)

        return sample_results

    def evaluate_full_test_set(self):
        """Evaluate the model on the entire test set"""
        print(f"\n=== Evaluating Full Test Set ({len(self.test_files)} images) ===")

        all_metrics = []
        failed_predictions = 0

        for i, img_file in enumerate(self.test_files):
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(self.test_files)} images...")

            result = self.predict_single_image(img_file)
            if result is not None and result['metrics']:
                all_metrics.append(result['metrics'])
                self.results.append(result)
            else:
                failed_predictions += 1

        if len(all_metrics) == 0:
            print("No valid predictions with ground truth found!")
            return None

        # Calculate overall statistics
        df_metrics = pd.DataFrame(all_metrics)
        overall_stats = {
            'mean': df_metrics.mean(),
            'std': df_metrics.std(),
            'median': df_metrics.median(),
            'min': df_metrics.min(),
            'max': df_metrics.max()
        }

        print(f"\n=== Overall Test Set Results ===")
        print(f"Successfully processed: {len(all_metrics)} images")
        print(f"Failed predictions: {failed_predictions}")
        print(f"\nOverall Metrics (Mean ± Std):")
        for metric in df_metrics.columns:
            mean_val = overall_stats['mean'][metric]
            std_val = overall_stats['std'][metric]
            print(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")

        return df_metrics, overall_stats

    def plot_metrics_distribution(self, df_metrics):
        """Plot distribution of metrics across test set"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Distribution of Metrics Across Test Set', fontsize=16, fontweight='bold')

        metrics = df_metrics.columns
        axes_flat = axes.flatten()

        for i, metric in enumerate(metrics):
            if i < len(axes_flat):
                # Histogram
                axes_flat[i].hist(df_metrics[metric], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes_flat[i].axvline(df_metrics[metric].mean(), color='red', linestyle='--',
                                     label=f'Mean: {df_metrics[metric].mean():.3f}')
                axes_flat[i].axvline(df_metrics[metric].median(), color='green', linestyle='--',
                                     label=f'Median: {df_metrics[metric].median():.3f}')
                axes_flat[i].set_title(f'{metric.capitalize()} Distribution')
                axes_flat[i].set_xlabel(metric.capitalize())
                axes_flat[i].set_ylabel('Frequency')
                axes_flat[i].legend()

        plt.tight_layout()
        plt.show()

    def create_summary_report(self, df_metrics, overall_stats):
        """Create a comprehensive summary report"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = f"model_evaluation_report_{timestamp}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Test Images Path: {self.test_img_path}\n")
            f.write(f"Test Masks Path: {self.test_mask_path}\n")
            f.write(f"Total Test Images: {len(self.test_files)}\n")
            f.write(f"Successfully Processed: {len(df_metrics)}\n\n")

            f.write("OVERALL STATISTICS\n")
            f.write("-" * 30 + "\n")
            for metric in df_metrics.columns:
                f.write(f"{metric.upper()}:\n")
                f.write(f"  Mean: {overall_stats['mean'][metric]:.6f}\n")
                f.write(f"  Std:  {overall_stats['std'][metric]:.6f}\n")
                f.write(f"  Min:  {overall_stats['min'][metric]:.6f}\n")
                f.write(f"  Max:  {overall_stats['max'][metric]:.6f}\n")
                f.write(f"  Median: {overall_stats['median'][metric]:.6f}\n\n")

            # Performance interpretation
            f.write("PERFORMANCE INTERPRETATION\n")
            f.write("-" * 30 + "\n")
            dice_mean = overall_stats['mean']['dice']
            jaccard_mean = overall_stats['mean']['jaccard']

            if dice_mean >= 0.9:
                f.write("Dice Score: EXCELLENT (>=0.90)\n")
            elif dice_mean >= 0.8:
                f.write("Dice Score: GOOD (0.80-0.89)\n")
            elif dice_mean >= 0.7:
                f.write("Dice Score: FAIR (0.70-0.79)\n")
            else:
                f.write("Dice Score: POOR (<0.70)\n")

            if jaccard_mean >= 0.8:
                f.write("IoU Score: EXCELLENT (>=0.80)\n")
            elif jaccard_mean >= 0.65:
                f.write("IoU Score: GOOD (0.65-0.79)\n")
            elif jaccard_mean >= 0.5:
                f.write("IoU Score: FAIR (0.50-0.64)\n")
            else:
                f.write("IoU Score: POOR (<0.50)\n")

        print(f"\nDetailed report saved to: {report_path}")
        return report_path


def main():
    """Main testing function"""
    print("=" * 60)
    print("DENTAL SEGMENTATION MODEL TESTING")
    print("=" * 60)

    # Initialize tester
    tester = ModelTester(MODEL_PATH, TEST_IMG_PATH, TEST_MSK_PATH)

    # Load model
    if not tester.load_model():
        print("Failed to load model. Exiting.")
        return

    # Get test files
    if not tester.get_test_files():
        print("No test files found. Exiting.")
        return

    # Test on sample images first
    print("\n" + "=" * 50)
    print("STEP 1: SAMPLE IMAGE TESTING")
    print("=" * 50)
    sample_results = tester.test_on_sample_images(num_samples=5)

    # Evaluate full test set
    print("\n" + "=" * 50)
    print("STEP 2: FULL TEST SET EVALUATION")
    print("=" * 50)
    df_metrics, overall_stats = tester.evaluate_full_test_set()

    if df_metrics is not None:
        # Plot metrics distribution
        print("\n" + "=" * 50)
        print("STEP 3: METRICS VISUALIZATION")
        print("=" * 50)
        tester.plot_metrics_distribution(df_metrics)

        # Create summary report
        print("\n" + "=" * 50)
        print("STEP 4: GENERATING REPORT")
        print("=" * 50)
        report_path = tester.create_summary_report(df_metrics, overall_stats)

        print("\n" + "=" * 60)
        print("TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Summary Report: {report_path}")
        print("Check the generated visualizations and report for detailed analysis.")

    return tester


if __name__ == "__main__":
    tester = main()