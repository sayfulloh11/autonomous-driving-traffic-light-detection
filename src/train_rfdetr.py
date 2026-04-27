# from rfdetr import RFDETRLarge

# def main():
#     model = RFDETRLarge()

#     model.train(
#         dataset_dir="data_samples/coco_dataset",
#         output_dir="results/rfdetr_runs",
#         epochs=30,
#         batch_size=8,
#         grad_accum_steps=4,
#         image_size=1024,
#         lr=5e-5,
#         early_stopping=True,
#         gradient_checkpointing=True,
#         strategy="ddp_find_unused_parameters_true",
#         devices="auto",
#     )

# if __name__ == "__main__":
#     main()


from rfdetr import RFDETRLarge

def main():
    model = RFDETRLarge()

    model.train(
        dataset_dir="data_samples/coco_dataset",
        output_dir="results/rfdetr_runs",
        epochs=30,
        batch_size=8,
        grad_accum_steps=4,
        lr=5e-5,
        image_size=1024,
        early_stopping=True,
        gradient_checkpointing=True,
        strategy="ddp_find_unused_parameters_true",
        devices="auto",
        resume="weights/last.ckpt",
    )

if __name__ == "__main__":
    main()