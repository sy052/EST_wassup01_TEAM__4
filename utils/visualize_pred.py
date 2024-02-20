def visualize_model_predictions(model, img_path, device, class_names):
    model.eval()

    img = Image.open(img_path)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        plt.imshow(img.cpu().data[0])
        plt.savefig('predicted_image.png')