# AI-Fractals Project TODO & Issues

## PROJECT PURPOSE & ARCHITECTURE

### Core Goal
Automatically generate high-quality fractal datasets for training GANs and other AI models.
This is a DATA GENERATION PIPELINE, not a manual curation tool.

### Pipeline Flow
1. **Parameter Generation**: Adaptive algorithms suggest fractal parameters
2. **Fractal Generation**: GPU-accelerated rendering (Mandelbrot/Julia)
3. **Quality Evaluation**: Metrics-based scoring (fractal dimension, complexity, visual properties)
4. **Selection**: Automatic acceptance based on quality thresholds
5. **Storage**: Save fractals with metadata for AI training

### Key Constraint
Must generate fractals FAST and AUTOMATICALLY - no manual intervention.
Quality must be determined by METRICS, not human labeling.

---

## CRITICAL ISSUES (HIGH PRIORITY)

### [CRITICAL] Issue 1: Process Management & Orphans
**Status**: BROKEN - Processes survive Ctrl+C and accumulate

**Problem**:
- KeyboardInterrupt catches Ctrl+C but doesn't terminate properly
- TensorFlow GPU context holds resources
- No cleanup handlers (atexit, signal, finally blocks)
- Process keeps running in background consuming CPU/GPU

**Root Cause**:
```python
# automatic_pipeline.py line 241
except KeyboardInterrupt:
    print("\n\nInterrupted by user. Saving progress...")
    break  # Only breaks loop, doesn't cleanup or exit
```

**Solution Needed**:
1. Add signal handlers (SIGINT, SIGTERM)
2. Use atexit for guaranteed cleanup
3. Add finally blocks to release GPU resources
4. Proper TensorFlow session cleanup: `tf.keras.backend.clear_session()`
5. Explicit `sys.exit(0)` after interrupt

**Files to Fix**:
- `src/ai_fractals/pipeline/automatic_pipeline.py`
- `scripts/generate_dataset.py`
- `scripts/generate_basic_fractals.py`

---

### [CRITICAL] Issue 2: Non-Square Images
**Status**: BROKEN - Fractals are rectangular, not square

**Problem**:
- Image shown: 400x600 pixels (aspect ratio distorted)
- Parameter ranges allow different width/height values:
  ```python
  'width': ParameterRange(400, 1200),
  'height': ParameterRange(400, 1200),
  ```
- Results in stretched/compressed fractals with incorrect proportions

**Impact**:
- AI models expect square inputs (256x256, 512x512, etc.)
- Distorted fractals lose mathematical properties
- Training data inconsistent for CNN/GAN

**Solution**:
1. Force square dimensions: `width = height`
2. Add aspect ratio validation
3. Update parameter sampler to enforce square constraint
4. Standard sizes: 512x512, 768x768, 1024x1024

**Files to Fix**:
- `src/ai_fractals/pipeline/parameter_sampler.py`
- `src/ai_fractals/generators/fractal_generators.py`

---

### [HIGH] Issue 3: Low Iteration Count
**Status**: SUBOPTIMAL - Max 512 iterations insufficient for detail

**Problem**:
- Current range: `max_iter: ParameterRange(128, 512)`
- 512 iterations show basic structure but miss fine detail
- Deep zoom regions need 1000+ iterations for complexity
- Paper emphasizes "multiscale complexity" - needs higher iter

**Impact**:
- Fractals lack intricate boundary details
- Missing self-similar patterns at small scales
- Less interesting for AI training

**Solution**:
1. Increase range: `max_iter: ParameterRange(512, 2048)`
2. Add zoom-depth parameter that scales iterations
3. Deeper zoom = higher iterations automatically
4. Balance: computation time vs visual quality

**Files to Fix**:
- `src/ai_fractals/pipeline/parameter_sampler.py`

---

## ARCHITECTURAL MISUNDERSTANDING - CORRECTED!

### The RIGHT CNN Approach: Parameter-to-Quality Predictor

**What We Should Do**:
Train CNN to predict: `f(parameters) -> quality_score`

**How It Works**:
1. Generate fractals with random parameters
2. Calculate quality metrics for each (fractal_dim, complexity, etc.)
3. Train CNN: Input = parameters (x_min, x_max, y_min, y_max, max_iter, etc.)
                Output = predicted quality score
4. Use trained CNN to quickly evaluate parameter combinations
5. Parameter search becomes MUCH faster (CNN prediction vs full generation + metrics)

**Why This Makes Sense**:
- CNN learns the "landscape" of good fractal parameters
- Can predict quality without generating the image
- Guides parameter search to promising regions
- 1000x faster than generating + evaluating each fractal
- Learns correlations: "zooming into (-0.7, 0.0) usually produces interesting fractals"

**Training Data**:
- Need 10,000+ parameter-quality pairs
- Can collect while generating dataset
- Save metadata: parameters + calculated metrics
- Train on this (param → metric) mapping

**Architecture**:
```
Input: [x_min, x_max, y_min, y_max, max_iter, ...]  # 5-10 values
Hidden: Dense(128) -> ReLU -> Dropout
        Dense(64) -> ReLU -> Dropout
        Dense(32) -> ReLU
Output: quality_score  # Single value 0-1
```

**Integration**:
- Replace Bayesian optimizer with CNN predictor
- Sample 1000 parameters, CNN predicts quality for all
- Generate only the top 10% most promising
- Huge speedup: evaluate 1000 params in 0.1 sec instead of 30 min

---

## CLEAN PIPELINE STRUCTURE

### Phase 1: Data Collection (Current)
**Script**: `generate_dataset.py`
**Purpose**: Generate fractals + collect parameter-quality data
**Output**:
- Images for GAN training
- Metadata JSON files with parameters + metrics

### Phase 2: Train Quality Predictor
**Script**: `train_quality_predictor.py` (NEW - need to create)
**Purpose**: Train CNN on parameters → quality mapping
**Input**: metadata/*.json files from Phase 1
**Output**: `models/quality_predictor.keras`

### Phase 3: Accelerated Generation
**Script**: `generate_with_predictor.py` (NEW - need to create)
**Purpose**: Use trained CNN to guide parameter search
**Speed**: 10-100x faster than Phase 1
**Output**: More high-quality fractals

### Phase 4: GAN Training
**Script**: `train_gan.py` (future)
**Purpose**: Train GAN on collected fractals
**Input**: All images from Phases 1-3

---

## ACTION PLAN

### Phase 1: Fix Critical Issues (DO THIS FIRST)

1. **Process Management** [COMPLETED ✓]
   - [x] Add signal handlers to automatic_pipeline.py
   - [x] Implement atexit cleanup function
   - [x] Add finally blocks for GPU cleanup
   - [x] Test Ctrl+C handling thoroughly
   - [x] Verify no orphan processes with `ps aux`

2. **Square Images** [COMPLETED ✓]
   - [x] Update parameter_sampler.py to force width=height
   - [x] Add validation in generate_fractal()
   - [x] Test with sample generation
   - [x] Update parameter ranges to standard sizes

3. **Higher Iterations** [COMPLETED ✓]
   - [x] Change max_iter range to (512, 2048)
   - [x] Test generation time impact
   - [x] Adjust based on performance

### Phase 2: Train Quality Predictor (NEXT PRIORITY)

4. **Create Training Script** [TODO]
   - [ ] Create `scripts/train_quality_predictor.py`
   - [ ] Load metadata from Phase 1 generation
   - [ ] Extract parameter vectors and quality scores
   - [ ] Train small fully-connected network
   - [ ] Save model as `models/quality_predictor.keras`

5. **Collect Training Data** [TODO]
   - [ ] Run Phase 1 with low threshold (0.2) to collect diverse data
   - [ ] Target: 10,000+ parameter-quality pairs
   - [ ] Ensure parameter space coverage
   - [ ] Validate data distribution

6. **Implement CNN Predictor** [TODO]
   - [ ] Simple architecture: Dense(128) -> Dense(64) -> Dense(1)
   - [ ] Input: normalized parameters [x_min, x_max, y_min, y_max, max_iter]
   - [ ] Output: predicted quality score
   - [ ] Fast inference: <0.0001 sec per prediction

### Phase 3: Accelerated Generation (AFTER PHASE 2)

7. **Create Accelerated Script** [TODO]
   - [ ] Create `scripts/generate_with_predictor.py`
   - [ ] Load trained quality predictor
   - [ ] Sample 1000 params, predict quality for all
   - [ ] Generate only top 10% (100 images)
   - [ ] 10-100x speedup over brute force

8. **Validate Predictor** [TODO]
   - [ ] Compare CNN predictions vs actual quality
   - [ ] Check if it finds interesting regions
   - [ ] Measure speedup (time to N accepted fractals)
   - [ ] Refine if needed

---

## OLD SECTIONS (keep for reference)

### Phase 2: Optimize Generation Pipeline (DEPRIORITIZED)
   - [ ] Analyze why acceptance rate is 0% at threshold 0.3
   - [ ] Review fractal_dimension calculation accuracy
   - [ ] Check if thresholds match actual score distributions
   - [ ] Adjust thresholds or improve metrics

5. **Parameter Search Tuning**
   - [ ] Verify optimizer integer conversion (already done)
   - [ ] Test Bayesian optimization convergence
   - [ ] Analyze which parameters produce high-quality fractals
   - [ ] Refine interesting_regions based on results

6. **Performance Optimization**
   - [ ] Profile GPU usage during generation
   - [ ] Optimize box-counting algorithm
   - [ ] Consider batch processing for evaluation
   - [ ] Target: >5 accepted fractals/minute

### Phase 3: Scale Up Production

7. **Generate Training Dataset**
   - [ ] Target: 10,000 high-quality fractals
   - [ ] Both Mandelbrot and Julia sets
   - [ ] Diverse parameter coverage
   - [ ] Save metadata for analysis

8. **Validation & Analysis**
   - [ ] Visualize parameter distributions
   - [ ] Check fractal diversity
   - [ ] Verify quality metrics correlate with visual interest
   - [ ] Prepare dataset documentation

### Phase 4: GAN Training Preparation

9. **Dataset Organization**
   - [ ] Structure: train/val/test splits
   - [ ] Create data loaders
   - [ ] Preprocessing pipeline
   - [ ] Metadata indexing

10. **GAN Architecture Design**
    - [ ] Review paper Section 7 (AI models)
    - [ ] Design generator architecture
    - [ ] Design discriminator architecture
    - [ ] Plan training strategy

---

## TECHNICAL DECISIONS LOG

### Why Sequential Processing?
- GPU context doesn't transfer across processes
- ProcessPoolExecutor caused hanging and orphans
- Single GPU process is fast enough (GPU parallelizes internally)
- Simpler code, easier debugging

### Why Adaptive Parameter Search?
- Random sampling: 72,000 attempts, 0 accepted (0.00%)
- Need intelligent exploration of parameter space
- Bayesian optimization balances exploration/exploitation
- Evolutionary algorithm learns from successful parameters

### Why Quality Thresholds?
- GAN training needs high-quality inputs
- Poor fractals produce poor GAN outputs
- Better to have 1000 excellent than 10000 mediocre
- Thresholds enforce quality bar

---

## CODE STYLE RULES

**DO NOT TOUCH** - Apply to all code:

1. **Docstrings**: ASCII only, no emojis
2. **Module docstrings**: Comprehensive, explain purpose
3. **Function/method docstrings**: One-line when possible
4. **Type hints**: Always use (makes docstrings shorter)
5. **Paper references**: Keep in comments where relevant
6. **Formatting**: Follow project conventions
