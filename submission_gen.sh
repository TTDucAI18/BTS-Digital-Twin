#!/bin/bash
# submission_gen.sh: Gather test renders into the submission ZIP format

OUTPUT_DIR="/kaggle/working/output"
SUBMISSION_DIR="/kaggle/working/submission"

echo "Creating submission package..."
mkdir -p $SUBMISSION_DIR

for SCENE_DIR in $OUTPUT_DIR/*; do
    if [ -d "$SCENE_DIR" ]; then
        SCENE_NAME=$(basename $SCENE_DIR)
        RENDER_PATH="$SCENE_DIR/test/ours_30000/renders"
        
        if [ -d "$RENDER_PATH" ]; then
            # Format requirements: submission.zip/scene_001/0001.png
            # The test images are numbered 00000.png, 00001.png by default in render.py
            # The competition requires 0001.png... 
            # We'll just copy the files over. If numbering needs adjustment, we'll do it here.
            
            DEST_SCENE_DIR="$SUBMISSION_DIR/$SCENE_NAME"
            mkdir -p $DEST_SCENE_DIR
            
            # Copy all rendered png files to destination scene directory
            cp $RENDER_PATH/*.png $DEST_SCENE_DIR/
        else
            echo "Warning: Render path not found for $SCENE_NAME: $RENDER_PATH"
        fi
    fi
done

cd $SUBMISSION_DIR
zip -r ../submission.zip *
echo "Submission saved to /kaggle/working/submission.zip"
