echo "========================================"
echo "Entering virtual environment if not already active..."
echo "========================================"
if [ -z "$VIRTUAL_ENV" ]; then
    # Try to activate the virtual environment
    if [ -f "../proj/Scripts/activate" ]; then
        echo "Activating virtual environment from ./proj/Scripts/activate"
        # shellcheck disable=SC1091
        source ../proj/Scripts/activate
    else
        echo "Virtual environment activation script not found at ./proj/Scripts/activate"
    fi
else
    echo "Virtual environment already active: $VIRTUAL_ENV"
fi

echo "========================================"
echo "Running: Feature Engineering Tests"
echo "========================================"
pytest test_feature_engineering.py -v -s
FEAT_STATUS=$?
if [ $FEAT_STATUS -eq 0 ]; then
    echo "Feature Engineering Tests PASSED"
else
    echo "Feature Engineering Tests FAILED"
fi

echo "========================================"
echo "Running: KMeans Tests"
echo "========================================"
pytest test_kmeans.py -v -s
KMEANS_STATUS=$?
if [ $KMEANS_STATUS -eq 0 ]; then
    echo "KMeans Tests PASSED"
else
    echo "KMeans Tests FAILED"
fi

if [ $FEAT_STATUS -eq 0 ] && [ $KMEANS_STATUS -eq 0 ]; then
    echo "\nALL TESTS PASSED\n"
else
    echo "\nSOME TESTS FAILED\n"
fi
# Do not exit with nonzero status, so the terminal stays open
