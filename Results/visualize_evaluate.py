#visualising the result

def visualize_predictions(model, dataset, device, index=0):
    """
    Plots original MRI image, True Mask, Predicted Probabilities, and Binary Mask Overlay.
    """
    model.eval()

    image, mask = dataset[index]
    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_input)
        pred = torch.sigmoid(output)

    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()
    binary_pred = (pred_np > 0.5).astype(np.float32)


    overlay = image_np.copy()

    overlay[binary_pred == 1] = [1.0, 0.0, 0.0]

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(image_np)
    plt.title("Original MRI Scan")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred_np, cmap='jet')
    plt.title("Model Confidence Map")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0))
    plt.title("Predicted Mask Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def calculate_metrics(pred, mask, threshold=0.5):
    """
    Calculates Dice and IoU scores for a single pair of prediction and mask.
    """
    pred = (pred > threshold).astype(np.float32)
    mask = mask.astype(np.float32)

    intersection = np.sum(pred * mask)
    total_pixels = np.sum(pred) + np.sum(mask)
    union = np.sum((pred + mask) > 0)

    smooth = 1e-8

    # Dice Score (F1-score equivalent)
    dice = (2.0 * intersection + smooth) / (total_pixels + smooth)
    # IoU Score (Jaccard Index)
    iou = (intersection + smooth) / (union + smooth)

    return dice, iou
  
#calculating dice score and iou
def evaluate_model(model, dataset, device):

    model.eval()
    dice_scores = []
    iou_scores = []

    for i in range(len(dataset)):
        image, mask = dataset[i]
        image_input = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_input)
            pred = torch.sigmoid(output)

        pred = pred.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()

        dice, iou = calculate_metrics(pred, mask)
        dice_scores.append(dice)
        iou_scores.append(iou)

    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)

    print(f"\n--- Validation Results ---")
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"Mean IoU (Jaccard Index): {mean_iou:.4f}")
    return mean_dice, mean_iou


visualize_predictions(model, test_dataset, device, index=5)

mean_dice, mean_iou = evaluate_model(model, test_dataset, device)



