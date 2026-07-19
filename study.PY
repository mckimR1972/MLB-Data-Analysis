import pandas as pd 
import oracledb as DB
import numpy as np
import seaborn as sns
import traceback
import matplotlib.pyplot as plt
import folium
import missingno as msno

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

non_ab_events = [
    "walk", "intent_walk", "hit_by_pitch",
    "sac_fly", "sac_bunt", "sac_fly_double_play",
    "catcher_interf" 
]
hit_for_run_events = ["single", "double", "triple", "home_run"]

def hit_distance_analysis(df:pd.DataFrame):
    df = df[
        (df['LAUNCH_SPEED'] >= 40) &
        (df['LAUNCH_ANGLE'] >= -30) & 
        (df['LAUNCH_ANGLE'] <= 55) &
        (df['HIT_DISTANCE_SC'] >= 10)].copy()
    feats = ['HIT_DISTANCE_SC', 'LAUNCH_SPEED', 'LAUNCH_ANGLE']
    df = df.dropna(subset=feats)

    # ===============================================

    X = df[['LAUNCH_SPEED', 'LAUNCH_ANGLE']]
    y = df['HIT_DISTANCE_SC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    degrees = range(1, 7)
    results = []

    for d in degrees:
        pnr = PolynomialFeatures(degree=d, include_bias=False)
        X_poly_train = pnr.fit_transform(X_train)
        X_poly_test = pnr.transform(X_test)

        lr = LinearRegression().fit(X_poly_train, y_train)
        y_hat = lr.predict(X_poly_test)

        results.append({
            'degree': d,
            'n_features': X_poly_train.shape[1],
            'train_r2': lr.score(X_poly_train, y_train),
            'test_r2': lr.score(X_poly_test, y_test),
            'mse': mean_squared_error(y_test, y_hat),
            'rmse': root_mean_squared_error(y_test, y_hat),
            'mae': mean_absolute_error(y_test, y_hat),
        })

        print(f"  degree={d} | features: {X_poly_train.shape[1]:>3} | "
            f"R²: {results[-1]['test_r2']:.4f} | "
            f"RMSE: {results[-1]['rmse']:.2f} | "
            f"MAE: {results[-1]['mae']:.2f}")

    res = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1) R² (Train vs Test)
    axes[0].plot(res['degree'], res['train_r2'], marker='o', color='#3266AD', linewidth=2, label='Train R²')
    axes[0].plot(res['degree'], res['test_r2'], marker='o', color='#D83E23', linewidth=2, label='Test R²')
    axes[0].fill_between(res['degree'], res['train_r2'], res['test_r2'], alpha=0.15, color='red')
    axes[0].set_xlabel('Polynomial Degree')
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² by Degree (Gap = Overfitting)')
    axes[0].legend()
    axes[0].set_xticks(res['degree'])
    for _, row in res.iterrows():
        axes[0].annotate(f"{row['test_r2']:.3f}", (row['degree'], row['test_r2']),
                        textcoords='offset points', xytext=(0, -15), ha='center', fontsize=9)

    # 2) RMSE & MAE
    axes[1].plot(res['degree'], res['rmse'], marker='s', color='#D83E23', linewidth=2, label='RMSE')
    axes[1].plot(res['degree'], res['mae'], marker='^', color='#F4A261', linewidth=2, label='MAE')
    axes[1].set_xlabel('Polynomial Degree')
    axes[1].set_ylabel('Error (ft)')
    axes[1].set_title('RMSE & MAE by Degree')
    axes[1].legend()
    axes[1].set_xticks(res['degree'])
    for _, row in res.iterrows():
        axes[1].annotate(f"{row['rmse']:.1f}", (row['degree'], row['rmse']),
                        textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9, color='#D83E23')
        axes[1].annotate(f"{row['mae']:.1f}", (row['degree'], row['mae']),
                        textcoords='offset points', xytext=(0, -15), ha='center', fontsize=9, color='#F4A261')

    # 3) 차수별 개선폭
    res['r2_gain'] = res['test_r2'].diff().fillna(0)
    colors = ['#3266AD' if g > 0.03 else '#F4A261' if g > 0 else '#D83E23' for g in res['r2_gain']]
    bars = axes[2].bar(res['degree'], res['r2_gain'], color=colors, edgecolor='white')
    for bar, val in zip(bars, res['r2_gain']):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'+{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('Polynomial Degree')
    axes[2].set_ylabel('ΔR² (vs previous degree)')
    axes[2].set_title('Marginal R² Gain per Degree')
    axes[2].set_xticks(res['degree'])
    axes[2].axhline(0, color='black', lw=0.8)

    fig.suptitle('Polynomial Regression: Degree Selection Analysis',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # --------------------------
    # 적합 데이터 수 : 52,626
    # --------------------------

    # from sklearn.preprocessing import PolynomialFeatures
    # from sklearn.linear_model import LinearRegression 
    # from sklearn.model_selection import train_test_split
    
    # pnr = PolynomialFeatures(degree=3, include_bias= False)
    # X = df[['LAUNCH_SPEED', 'LAUNCH_ANGLE']] 
    # y = df['HIT_DISTANCE_SC']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42) 

    # X_poly_train = pnr.fit_transform(X_train)
    # X_poly_test = pnr.transform(X_test)

    # lr = LinearRegression().fit(X_poly_train, y_train)
    # print(lr.score(X_poly_test, y_test))
    # print(lr.coef_)

    # from mpl_toolkits.mplot3d import Axes3D

    # # ── 예측 표면 생성 ──
    # speed_range = np.linspace(X_train['LAUNCH_SPEED'].min(), X_train['LAUNCH_SPEED'].max(), 80)
    # angle_range = np.linspace(X_train['LAUNCH_ANGLE'].min(), X_train['LAUNCH_ANGLE'].max(), 80)
    
    # S, A = np.meshgrid(speed_range, angle_range)

    # grid = pd.DataFrame({'LAUNCH_SPEED': S.ravel(), 'LAUNCH_ANGLE': A.ravel()})
    # Z = np.clip(lr.predict(pnr.transform(grid)), 0, 500).reshape(S.shape)

    # fig = plt.figure(figsize=(14, 9))
    # ax = fig.add_subplot(111, projection='3d')

    # # 예측 표면
    # ax.plot_surface(S, A, Z, cmap='RdYlBu_r', alpha=0.6, edgecolor='none')

    # # 실제 데이터 (테스트셋)
    # ax.scatter(X_test['LAUNCH_SPEED'], X_test['LAUNCH_ANGLE'], y_test,
    #         s=5, alpha=0.4, color='black', label='Actual')

    # ax.set_xlabel('Exit Velocity (mph)')
    # ax.set_ylabel('Launch Angle (°)')
    # ax.set_zlabel('Distance (ft)')
    # ax.set_title('Polynomial Regression Surface (degree=3)', fontsize=14, fontweight='bold')
    # ax.legend()

    # # 초기 시점 각도 조절
    # ax.view_init(elev=25, azim=135)

    # plt.tight_layout()
    # plt.show()
    
def pfx_z_analysis(df:pd.DataFrame):
    # 불완전 분석 
    # PITCH_TYPE 분리시켜서 변수 재분석 필요. 
    # 실제로 PITCH_TYPE 분리시 결정계수 변화폭이 너무 큼
    # 1. 우투수 전체 데이터 분포 비교 
    df['SPIN_AXIS_RAD'] = np.radians(df['SPIN_AXIS'])
    df['SIN_AXIS'] = np.sin(df['SPIN_AXIS_RAD'])
    df['COS_AXIS'] = np.cos(df['SPIN_AXIS_RAD'])

    features = ['VZ0', 'AY', 'SIN_AXIS', 'COS_AXIS', 'RELEASE_SPEED', 'RELEASE_SPIN_RATE']
    df = df.dropna(subset= features + ['PFX_Z'])
    df = df[df['P_THROWS'] == 'R']

    # g = sns.PairGrid(df[features])
    # g.map_diag(plt.hist, bins=60, color='#3266AD', edgecolor='white')
    # g.map_offdiag(plt.scatter, s=1, alpha=0.05, edgecolors='none', rasterized=True)

    # plt.show()
    
    #2. 랜덤 포레스트로 변수 중요도 파악 
    X = df[features]
    y = df['PFX_Z']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    importance = pd.Series(rf.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=True)   

    print(f"Random Forest R²: {rf.score(X_test, y_test):.3f}\n")
    print("변수 중요도:")
    for col, imp in importance.items():
        print(f"  {col:>20}: {imp:.3f}")

    importance.plot(kind='barh', color='#3266AD')
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.show() 


    # pitch_types = ['FF', 'SL', 'CH', 'CU', 'FC', 'SI']

    # for pt in pitch_types:
    #     pt_df = df[df['PITCH_TYPE'] == pt].copy()
    #     if len(pt_df) < 500:
    #         continue
        
    #     features = ['COS_AXIS', 'SIN_AXIS', 'RELEASE_SPEED', 'RELEASE_SPIN_RATE']
    #     X = pt_df[features]
    #     y = pt_df['PFX_Z']
        
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
    #     rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    #     rf.fit(X_train, y_train)
        
    #     pt_df['EXPECTED_PFX_Z'] = rf.predict(pt_df[features])
    #     pt_df['SPIN_EFFICIENCY'] = pt_df['PFX_Z'] - pt_df['EXPECTED_PFX_Z']
        
    #     print(f"\n{pt}: R²={rf.score(X_test, y_test):.3f} (n={len(pt_df):,})")


    # # 3. 자동 변수 선택 - Lasso
    from sklearn.linear_model import LassoCV 
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y_train)
    print("Lasso가 선택한 변수 : ")
    for col, coef in zip(X.columns, lasso.coef_):
        status = "유지" if abs(coef) > 0.01 else "제거"
        print(f"{col:>20}: {coef:>+.3f}  {status}")

    sns.heatmap(data=df[features].corr(), annot=True, cmap='RdBu', cbar=True, fmt=".3f") 
    plt.show()   

    # # 4. VIF 계산 — 다중공선성 확인
    from sklearn.linear_model import LinearRegression
    vif_df = df[features].dropna()

    print("\n" + "=" * 50)
    print("VIF (Variance Inflation Factor)")
    print("=" * 50)
    for col in features:
        others = [c for c in features if c != col]
        r2 = LinearRegression().fit(vif_df[others], vif_df[col]).score(vif_df[others], vif_df[col])
        vif = 1 / (1 - r2) if r2 < 1 else float('inf')
        flag = "⚠ 위험" if vif >= 10 else "주의" if vif >= 5 else "양호"
        print(f"  {col:>20}: VIF = {vif:>8.2f}  ({flag})")

    # # 5. PFX_Z 모델링 
    degree = np.arange(1, 6)
    result = []

    for d in degree:
        pnr = PolynomialFeatures(degree=d, include_bias=False)
        X_poly_train = pnr.fit_transform(X_train)
        X_poly_test = pnr.transform(X_test)

        lr = LinearRegression().fit(X_poly_train, y_train)

        result.append(
            {
              'degree' : d,
              'features' : X_poly_train.shape[1], 
             'train_r2' : lr.score(X_poly_train, y_train),
             'test_r2': lr.score(X_poly_test, y_test),
             'mae' : mean_absolute_error(y_test, lr.predict(X_poly_test)),
             'mse' : mean_squared_error(y_test, lr.predict(X_poly_test)),
             'rmse' : root_mean_squared_error(y_test, lr.predict(X_poly_test))
            }
        )
    res = pd.DataFrame(data=result)

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))   
    
    axes[0].plot(res['degree'], res['train_r2'], marker='o', color='blue', linewidth=2, label='TRAIN_R2')
    axes[0].plot(res['degree'], res['test_r2'], marker='o', color='Red', linewidth=2, label='TEST_R2')
    axes[0].set_xlabel('Polynomial Degree')
    axes[0].set_ylabel('R2 (Expression Rate)')
    axes[0].set_title('R2 For Train & Test by Degree')
    axes[0].legend()
    axes[0].set_xticks(res['degree'])
    for _, row in res.iterrows():
        axes[0].annotate(f"{row['train_r2']:.3f}", (row['degree'], row['train_r2']),
                        textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9, color='black')
        axes[0].annotate(f"{row['test_r2']:.3f}", (row['degree'], row['test_r2']),
                        textcoords='offset points', xytext=(0, -15), ha='center', fontsize=9, color='black')

    # 2) RMSE & MAE
    axes[1].plot(res['degree'], res['rmse'], marker='s', color='#D83E23', linewidth=2, label='RMSE')
    axes[1].plot(res['degree'], res['mae'], marker='^', color='#F4A261', linewidth=2, label='MAE')
    axes[1].set_xlabel('Polynomial Degree')
    axes[1].set_ylabel('Error (ft)')
    axes[1].set_title('RMSE & MAE by Degree')
    axes[1].legend()
    axes[1].set_xticks(res['degree'])
    for _, row in res.iterrows():
        axes[1].annotate(f"{row['rmse']:.3f}", (row['degree'], row['rmse']),
                        textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9, color='#D83E23')
        axes[1].annotate(f"{row['mae']:.3f}", (row['degree'], row['mae']),
                        textcoords='offset points', xytext=(0, -15), ha='center', fontsize=9, color='#F4A261')

    # 3) 차수별 개선폭
    res['r2_gain'] = res['test_r2'].diff().fillna(0)
    colors = ['#3266AD' if g > 0.03 else '#F4A261' if g > 0 else '#D83E23' for g in res['r2_gain']]
    bars = axes[2].bar(res['degree'], res['r2_gain'], color=colors, edgecolor='white')
    for bar, val in zip(bars, res['r2_gain']):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'+{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('Polynomial Degree')
    axes[2].set_ylabel('ΔR² (vs previous degree)')
    axes[2].set_title('Marginal R² Gain per Degree')
    axes[2].set_xticks(res['degree'])
    axes[2].axhline(0, color='black', lw=0.8)

    fig.suptitle('Polynomial Regression: Degree Selection Analysis For PFX_Z',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 6. 모델링 & 스핀 효율 top 10 투수 분석 
    pnr = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_train = pnr.fit_transform(X_train)
    X_poly_test = pnr.transform(X_test)

    lr = LinearRegression().fit(X_poly_train, y_train)

    df['EXPECTED_PFX_Z'] = lr.predict(pnr.transform(df[features]))
    df['SPIN_EFFICIENCY'] = df['PFX_Z'] - df['EXPECTED_PFX_Z']

    pitcher_eff = df.groupby('PITCHER').agg(
        spin_eff=('SPIN_EFFICIENCY', 'mean'),
        spin_eff_std=('SPIN_EFFICIENCY', 'std'),
        avg_pfx_z=('PFX_Z', 'mean'),
        avg_spin=('RELEASE_SPIN_RATE', 'mean'),
        avg_axis=('SPIN_AXIS', 'mean'),
        n=('PFX_Z', 'count')
    ).reset_index()
    pitcher_eff = pitcher_eff[pitcher_eff['n'] >= 200]
    pitcher_eff = pitcher_eff.sort_values('spin_eff', ascending=False)

    print("\n=== 스핀 효율 TOP 10 ===")
    print(pitcher_eff.head(10).to_string(index=False))
    print("\n=== 스핀 효율 BOTTOM 10 ===")
    print(pitcher_eff.tail(10).to_string(index=False))    
    
def pitch_type_classification(df:pd.DataFrame):
    features = ['SPIN_AXIS', 'AX', 'AY', 'AZ', 'RELEASE_SPEED']
    df = df.dropna(subset=features + ['PITCH_TYPE'])
    df = df[~(df['PITCH_TYPE'].isin(['PO', 'UN']))]
    df = df[df['P_THROWS'] == 'R']

    condlist=[
        df["PITCH_TYPE"].isin(["FF", "SI", "FA", "FO"]),
        df["PITCH_TYPE"].isin(["SL", "SV", "ST", "FC", "SC"]),
        df["PITCH_TYPE"].isin(["CU", "KC", "CS", "EP"]),
        df["PITCH_TYPE"].isin(["CH", "FS", "KN"]),
    ]
    choicelist=[
        'FASTBALL', 'BREAKING-G', 'BREAKING-D', 'OFFSPEED'
    ]

    df['PITCH_GROUP'] = np.select(condlist, choicelist, default='UNKNOWN')
    df['CLASS'] = np.where(
        df['PITCH_GROUP'] == 'OFFSPEED', 1, 0
    )

    X = df[features]
    y = df['CLASS'] 

    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split 
    from sklearn.neighbors import KNeighborsClassifier

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_hat = knn.predict(X_test)

    from sklearn import metrics 

    knn_matrix = metrics.confusion_matrix(y_test, y_hat)
    knn_report = metrics.classification_report(y_test, y_hat)

    print(knn_matrix)
    print(knn_report)

    from sklearn.metrics import balanced_accuracy_score, f1_score

    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_hat))
    print("F1 (macro):", f1_score(y_test, y_hat, average='macro'))   

    from sklearn.decomposition import PCA
    from sklearn.svm import SVC 

    sample = df.sample(n=10000, random_state=42)
    X_scaled = scaler.fit_transform(sample[features])
    y = sample['CLASS'].values

    pca = PCA(n_components=2)
    X_pca= pca.fit_transform(X_scaled)

    svc = SVC(kernel='rbf')
    svc.fit(X_pca, y)

    y_hat = svc.predict(X_pca)
    report = metrics.classification_report(y, y_hat)

    print(report)

    xx, yy = np.meshgrid(
        np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 300),
        np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 300)
    )
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='blue', s=5, alpha=0.3)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='red', s=5, alpha=0.3)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('SVC Decision Boundary (PCA)')
    plt.show()

    from sklearn import tree

    tree_model = tree.DecisionTreeClassifier(criterion='gini', max_depth=10)
    tree_model.fit(X_train, y_train)

    y_hat = tree_model.predict(X_test)
    tree_matrix = metrics.confusion_matrix(y_test, y_hat)
    tree_report = metrics.classification_report(y_test, y_hat)

    print(tree_matrix)
    print(tree_report)

    importances = tree_model.feature_importances_
    idx = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    plt.barh([features[i] for i in idx], importances[idx])
    plt.xlabel('Feature Importance (Gini)')
    plt.title('Decision Tree Feature Importance')
    plt.show()

    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        tree_model, X_test, y_test,
        n_repeats=10,         # 10번 반복해서 평균
        random_state=42
    )

    idx = np.argsort(result.importances_mean)
    plt.figure(figsize=(8, 5))
    plt.barh(
        [features[i] for i in idx],
        result.importances_mean[idx]
    )
    plt.xlabel('Permutation Importance (accuracy drop)')
    plt.title('Permutation Feature Importance')
    plt.show()

def boosting(df: pd.DataFrame) -> GradientBoostingClassifier:
    features = ['SPIN_AXIS', 'AX', 'AY', 'AZ', 'RELEASE_SPEED']
    df = df.dropna(subset=features + ['PITCH_TYPE'])
    df = df[~(df['PITCH_TYPE'].isin(['PO', 'UN']))]
    df = df[df['P_THROWS'] == 'R']

    condlist=[
        df["PITCH_TYPE"].isin(["FF", "SI", "FA", "FO"]),
        df["PITCH_TYPE"].isin(["SL", "SV", "ST", "FC", "SC"]),
        df["PITCH_TYPE"].isin(["CU", "KC", "CS", "EP"]),
        df["PITCH_TYPE"].isin(["CH", "FS", "KN"]),
    ]
    choicelist=[
        'FASTBALL', 'BREAKING-G', 'BREAKING-D', 'OFFSPEED'
    ]

    df['PITCH_GROUP'] = np.select(condlist, choicelist, default='UNKNOWN')
    df['CLASS'] = np.where(
        df['PITCH_GROUP'] == 'OFFSPEED', 1, 0
    )

    X = df[features]
    y = df['CLASS'] 

    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn import tree 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    dt = tree.DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    # AdaBoost 
    # ada = AdaBoostClassifier(
    #     estimator= dt,
    #     n_estimators=100,
    #     learning_rate=0.1,
    #     random_state=42 
    # )

    # ada.fit(X_train, y_train)
    # y_hat = ada.predict(X_test)

    # ada_matrix = metrics.confusion_matrix(y_test, y_hat)
    # ada_report = metrics.classification_report(y_test, y_hat)

    # print(f"혼동 행렬 표 \n{ada_matrix}\n")
    # print(f"AdaBoosting 리포트\n{ada_report}")
    # print("Train:", ada.score(X_train, y_train))
    # print("Test:", ada.score(X_test, y_test))
    # # 차이가 0.02 이상이면 과적합 의심

    # GradientBoost => 모델 내부적으로 의사결정 트리  
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_hat = gb.predict(X_test)

    print(metrics.confusion_matrix(y_test, y_hat))
    print(metrics.classification_report(y_test, y_hat))

    print(f"Train: {round(gb.score(X_train, y_train), 3)}")
    print(f"Test: {round(gb.score(X_test, y_test), 3)}")

    return gb 

def Masson_Miller(df:pd.DataFrame): 
    mm_origin = df[df['PITCHER'] == 695243]
    mm_origin['SEASON'] = pd.to_datetime(mm_origin['GAME_DATE']).dt.year
    mm_origin = mm_origin.sort_values(by=['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'], ascending=[True, True, True]).reset_index(drop=True)
    
    # =======================================
    # 1. 구종 통게
    # =======================================
    mm = mm_origin.copy()

    event = mm[mm['PITCH_EVENTS'].notna()]
    event['IS_AB'] = np.where(
        (event['PITCH_EVENTS'].notna()) & ~(event['PITCH_EVENTS'].isin([non_ab_events])),
        1, 0
    )
    event['IS_HIT_FOR_RUN'] = np.where(
        event['PITCH_EVENTS'].isin(hit_for_run_events), 1, 0
    )
    condlist = [
        event['PITCH_EVENTS'] == 'single',
        event['PITCH_EVENTS'] == 'double',
        event['PITCH_EVENTS'] == 'triple',
        event['PITCH_EVENTS'] == 'home_run'
    ]
    event['HIT_FOR_BASE'] = np.select(condlist, [1,2,3,4], default=0)

    event = event.groupby(['PITCH_TYPE', 'SEASON']).agg(
        AB = ('IS_AB', 'sum'),
        HIT_FOR_RUN=('IS_HIT_FOR_RUN', 'sum'),
        RUN_BASE = ('HIT_FOR_BASE', 'sum')
    )
    event['OBP'] = round(event['HIT_FOR_RUN'] / event['AB'], 3)
    event['OSLG'] = round(event['RUN_BASE'] / event['AB'], 3)

    mm = mm.groupby(['PITCH_TYPE', 'SEASON']).agg(
            COUNT = ('RELEASE_SPEED', 'count'),
            SPEED_AVG = ('RELEASE_SPEED', 'mean'),
    )
    mm['SPEED_AVG'] = round(mm['SPEED_AVG'], 1)
    mm['RATIO'] = round(mm['COUNT'] / mm.groupby('SEASON')['COUNT'].transform('sum') * 100, 1).astype(str) + "%"
    res = pd.merge(
        left=mm, right=event, how='inner',
        left_index=True, right_index=True
    )

    print(res)
    mask = (mm_origin['PITCH_TYPE'] == 'FF') & (mm_origin['SEASON'] == 2025)
    subset = mm_origin[mask]

    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax, year, title in zip(axes, [2025, 2026], ['2025 Slider', '2026 Slider']):
        subset = mm_origin[
            (mm_origin['PITCH_TYPE'] == 'SL') & (mm_origin['SEASON'] == year)
        ].dropna(subset=['PLATE_X', 'PLATE_Z'])

        sns.kdeplot(
            data=subset,
            x='PLATE_X', y='PLATE_Z',
            cmap='hot_r',          # 빨강→노랑→흰색 (이미지와 유사)
            fill=True,
            levels=100,             # 등고선 수를 높여서 부드럽게
            thresh=0.2,           # 밀도 낮은 외곽을 자연스럽게 페이드
            ax=ax,
            alpha=0.9,
        )

        # 스트라이크존 사각형 (피트 단위)
        sz = Rectangle(
            (-0.833, 1.5), 1.666, 2.0,
            fill=False, edgecolor='black', linewidth=2, zorder=10
        )
        ax.add_patch(sz)

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
        ax.set_xlabel('PLATE_X (CAPCHER VIEW)')
        ax.set_ylabel('PLATE_Z')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    mm = mm_origin.copy()
    mm['PREV_PITCH_TYPE'] = mm.groupby(['GAME_PK','AT_BAT_NUMBER'])['PITCH_TYPE'].shift(1)
    transition = pd.crosstab(
        mm['PREV_PITCH_TYPE'],
        mm['PITCH_TYPE'],
        normalize='index'
    )
    print(transition)

def calc_decision_point(df:pd.DataFrame):
    """
    스윙 결정 시점(t=0.25초)에서의 보정된 x, y, z 좌표 계산
    
    Parameters:
        df: PITCH_DESCRIPTION, VX0, VY0, VZ0, AX, AY, AZ, 
            RELEASE_POS_X, RELEASE_POS_Z, PLATE_X, PLATE_Z 컬럼 필요
        t_decision: 스윙 결정 시점 (기본 0.25초)
    
    Returns:
        DataFrame with columns: X_AT_DECISION, Y_AT_DECISION, Z_AT_DECISION
    """
    t_decision=0.25

    # ── 홈플레이트 도달 시간 계산 ──
    # y(t) = RELEASE_POS_Y + VY0*t + 0.5*AY*t² = 0
    a = 0.5 * df['AY']
    b = df['VY0']
    c = df['RELEASE_POS_Y'] - 1.41

    discriminant = b**2 - 4*a*c
    t_plate = (-b - np.sqrt(discriminant)) / (2 * a)

    # ── 궤적 공식으로 예측한 홈플레이트 좌표 ──
    pred_plate_x = df['RELEASE_POS_X'] + df['VX0'] * t_plate + 0.5 * df['AX'] * t_plate**2
    pred_plate_z = df['RELEASE_POS_Z'] + df['VZ0'] * t_plate + 0.5 * df['AZ'] * t_plate**2

    # ── 보정 계수 ──
    error_x = df['PLATE_X'] - pred_plate_x
    error_z = df['PLATE_Z'] - pred_plate_z
    correction_ratio = t_decision / t_plate

    # ── 결정 시점 원시 좌표 ──
    raw_x = df['RELEASE_POS_X'] + df['VX0'] * t_decision + 0.5 * df['AX'] * t_decision**2
    raw_z = df['RELEASE_POS_Z'] + df['VZ0'] * t_decision + 0.5 * df['AZ'] * t_decision**2
    raw_y = df['RELEASE_POS_Y'] + df['VY0'] * t_decision + 0.5 * df['AY'] * t_decision**2

    # ── 보정 적용 ──
    result = pd.DataFrame({
        'X_AT_DECISION': raw_x + error_x * correction_ratio,
        'Y_AT_DECISION': raw_y,
        'Z_AT_DECISION': raw_z + error_z * correction_ratio,
    }, index=df.index)

    return result

# 미사용 함수 
def calc_trajectory(row, n_points=10):
    """
    한 투구의 궤적을 n개 시점으로 샘플링 (보정 포함)
    
    Parameters:
        row: Series — VX0, VY0, VZ0, AX, AY, AZ,
             RELEASE_POS_X, RELEASE_POS_Y, RELEASE_POS_Z,
             PLATE_X, PLATE_Z 필요
        n_points: 샘플링할 시점 수
    
    Returns:
        numpy array, shape (n_points, 2) — 각 시점의 [x, z]
    """
    # ── 홈플레이트 도달 시간 계산 ──
    a = 0.5 * row['AY']
    b = row['VY0']
    c = row['RELEASE_POS_Y'] - 1.41

    discriminant = b**2 - 4*a*c
    t_plate = (-b - np.sqrt(discriminant)) / (2 * a)

    # ── 궤적 공식으로 예측한 홈플레이트 좌표 ──
    pred_plate_x = row['RELEASE_POS_X'] + row['VX0'] * t_plate + 0.5 * row['AX'] * t_plate**2
    pred_plate_z = row['RELEASE_POS_Z'] + row['VZ0'] * t_plate + 0.5 * row['AZ'] * t_plate**2

    # ── 보정 계수 ──
    error_x = row['PLATE_X'] - pred_plate_x
    error_z = row['PLATE_Z'] - pred_plate_z

    # ── n개 시점에서 보정된 좌표 계산 ──
    times = np.linspace(0, t_plate, n_points)
    trajectory = np.zeros((n_points, 2))

    for i, t in enumerate(times):
        ratio = t / t_plate
        trajectory[i, 0] = row['RELEASE_POS_X'] + row['VX0'] * t + 0.5 * row['AX'] * t**2 + error_x * ratio
        trajectory[i, 1] = row['RELEASE_POS_Z'] + row['VZ0'] * t + 0.5 * row['AZ'] * t**2 + error_z * ratio

    return trajectory

def build_sliding_data(df):
        MAX_PITCH = 15
        all_rows = []

        for (gk, ab), group in df.groupby(['GAME_PK', 'AT_BAT_NUMBER']):
            group = group.sort_values('PITCH_NUMBER')

            for t in range(2, len(group) + 1):
                history = group.iloc[:t - 1]
                target = group.iloc[t - 1]

                row = {}

                # ── 이전 투구들의 원시 값 + 패딩 ──
                for i in range(MAX_PITCH - 1):
                    if i < len(history):
                        pitch = history.iloc[i]
                        for col in pitch_features:
                            row[f'p{i+1}_{col}'] = pitch[col]
                        row[f'p{i+1}_EXISTS'] = 1
                    else:
                        for col in pitch_features:
                            row[f'p{i+1}_{col}'] = 0
                        row[f'p{i+1}_EXISTS'] = 0

                # ── 타자 반응 ──
                for i in range(MAX_PITCH - 1):
                    if i < len(history):
                        row[f'p{i+1}_PITCH_DESCRIPTION'] = history.iloc[i]['PITCH_DESCRIPTION']
                    else:
                        row[f'p{i+1}_PITCH_DESCRIPTION'] = 'PAD'

                # ── 이전 투구의 구종 라벨 ──
                for i in range(MAX_PITCH - 1):
                    if i < len(history):
                        row[f'p{i+1}_IS_FF'] = 1 if history.iloc[i]['PITCH_TYPE'] == 'FF' else 0
                        row[f'p{i+1}_IS_SL'] = 1 if history.iloc[i]['PITCH_TYPE'] == 'SL' else 0
                    else:
                        row[f'p{i+1}_IS_FF'] = 0
                        row[f'p{i+1}_IS_SL'] = 0

                # ── 상황 변수 ──
                row['ACTUAL_LENGTH'] = len(history)
                row['BALLS'] = target['BALLS']
                row['STRIKES'] = target['STRIKES']
                row['STAND'] = 1 if target['STAND'] == 'R' else 0

                # ── 타석 식별 키 (GroupKFold용) ──
                row['GAME_AB_KEY'] = f"{gk}_{ab}"

                # ── 예측 대상 ──

                row['target_TYPE'] = target['PITCH_TYPE']
                row['target_PLATE_X'] = target['PLATE_X']
                row['target_PLATE_Z'] = target['PLATE_Z']
                row['target_SPEED'] = target['RELEASE_SPEED']
                row['target_PFX_X'] = target['PFX_X']
                row['target_PFX_Z'] = target['PFX_Z']
                row['target_AX'] = target['AX']
                row['target_AZ'] = target['AZ']
                row['target_RESULT'] = target['PITCH_DESCRIPTION']
                row['target_SZ_TOP'] = 1.5
                row['target_SZ_BOT'] = 3.5

                all_rows.append(row)

        return pd.DataFrame(all_rows)

def my_model(df: pd.DataFrame):
    # 데이터 전처리 
    pitch_features = [
        'RELEASE_SPEED', 'AX', 'AZ',
        'PFX_X', 'PFX_Z',
        'RELEASE_SPIN_RATE', 
        'X_AT_DECISION', 'Y_AT_DECISION', 'Z_AT_DECISION'
    ]

    rhp = df[(df['P_THROWS'] == 'R') & (df['PITCH_TYPE'].isin(['FF', 'SL']))]
    rhp = rhp.sort_values(['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])
    rhp['DID_SWING'] = np.where(
        rhp['PITCH_DESCRIPTION'].isin(
            ['swinging_strike', 'swinging_strike_blocked', 'foul',
             'foul_tip', 'foul_bunt', 'hit_into_play']
        ), 1, 0
    )

    col = calc_decision_point(df)
    rhp = pd.concat([rhp, col], axis=1)
    rhp = rhp.dropna(subset=pitch_features)

    # ── PREV 원시값 ──
    group = rhp.groupby(['GAME_PK', 'AT_BAT_NUMBER'])
    prev_cols = ['RELEASE_SPEED', 'AX', 'AZ', 'X_AT_DECISION', 'Z_AT_DECISION', 'PLATE_X', 'PLATE_Z']
    prev_df = group[prev_cols].shift(1).add_prefix('PREV_')
    rhp = pd.concat([rhp, prev_df], axis=1)

    # ── 궤적 괴리 비율 ──
    rhp['MOVE_RATE'] = (rhp['PLATE_X'] - rhp['X_AT_DECISION']) / rhp['X_AT_DECISION']
    rhp['DROP_RATE'] = (rhp['Z_AT_DECISION'] - rhp['PLATE_Z']) / rhp['Z_AT_DECISION']

    # ── 직전 대비 변화량 ──
    rhp['X_AT_DECISION_DIFF'] = rhp['X_AT_DECISION'] - rhp['PREV_X_AT_DECISION']
    rhp['Z_AT_DECISION_DIFF'] = rhp['Z_AT_DECISION'] - rhp['PREV_Z_AT_DECISION']

    # ── NaN 처리: PREV는 전체 평균으로, DIFF는 0으로 ──
    prev_fill_cols = ['PREV_RELEASE_SPEED', 'PREV_AX', 'PREV_AZ',
                      'PREV_X_AT_DECISION', 'PREV_Z_AT_DECISION',
                      'PREV_PLATE_X', 'PREV_PLATE_Z']
    for c in prev_fill_cols:
        original = c.replace('PREV_', '')
        rhp[c] = rhp[c].fillna(rhp[original].mean())

    rhp['X_AT_DECISION_DIFF'] = rhp['X_AT_DECISION_DIFF'].fillna(0)
    rhp['Z_AT_DECISION_DIFF'] = rhp['Z_AT_DECISION_DIFF'].fillna(0)
    rhp['MOVE_RATE'] = rhp['MOVE_RATE'].fillna(0)
    rhp['DROP_RATE'] = rhp['DROP_RATE'].fillna(0)

    # ── 존 안/밖 판별 (타자 성향 계산용) ──
    DZ_TOP = 4.69
    DZ_BOT = 3.76
    DZ_EDGE = 0.63

    rhp['IN_ZONE_AT_DECISION'] = (
        (abs(rhp['X_AT_DECISION']) <= DZ_EDGE) &
        (rhp['Z_AT_DECISION'] >= DZ_BOT) &
        (rhp['Z_AT_DECISION'] <= DZ_TOP)
    ).astype(int)
    
    # 정렬 (타자별 시간순)
    rhp = rhp.sort_values(['BATTER', 'GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])

    # ── 1. 타자 누적 스윙율 (과거 이력만) ──
    rhp['BATTER_CUM_SWING'] = (
        rhp.groupby('BATTER')['DID_SWING']
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # ── 2. 존 안/밖 별 스윙율 ──
    # 존 안 공에 대한 스윙율 (공격성)
    rhp['IN_ZONE_SWING'] = rhp['IN_ZONE_AT_DECISION'] * rhp['DID_SWING']
    rhp['BATTER_ZONE_SWING'] = (
        rhp.groupby('BATTER')['IN_ZONE_SWING']
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # 존 밖 공에 대한 스윙율 (선구안)
    rhp['OUT_ZONE_SWING'] = (1 - rhp['IN_ZONE_AT_DECISION']) * rhp['DID_SWING']
    rhp['BATTER_CHASE_RATE'] = (
        rhp.groupby('BATTER')['OUT_ZONE_SWING']
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # ── 3. 해당 타석 내 스윙율 (현재 타석의 적극성) ──
    rhp = rhp.sort_values(['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])
    rhp['AB_CUM_SWING'] = (
        rhp.groupby(['GAME_PK', 'AT_BAT_NUMBER'])['DID_SWING']
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # ── 4. 타자의 누적 투구 수 (경험치) ──
    rhp = rhp.sort_values(['BATTER', 'GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])
    rhp['BATTER_PITCH_COUNT'] = rhp.groupby('BATTER').cumcount()

    # ── NaN 처리 ──
    global_swing = rhp['DID_SWING'].mean()
    rhp['BATTER_CUM_SWING'] = rhp['BATTER_CUM_SWING'].fillna(global_swing)
    rhp['BATTER_ZONE_SWING'] = rhp['BATTER_ZONE_SWING'].fillna(global_swing)
    rhp['BATTER_CHASE_RATE'] = rhp['BATTER_CHASE_RATE'].fillna(global_swing)
    rhp['AB_CUM_SWING'] = rhp['AB_CUM_SWING'].fillna(global_swing)

    # 중간 계산용 컬럼 제거
    rhp = rhp.drop(columns=['IN_ZONE_SWING', 'OUT_ZONE_SWING', 'PLATE_X', 'PLATE_Z'])

    # ── 최종 feature 리스트 ──
    model_features = [
        'RELEASE_SPEED', 'AX', 'AZ',
        'X_AT_DECISION', 'Z_AT_DECISION',
        'MOVE_RATE', 'DROP_RATE',
        'X_AT_DECISION_DIFF', 'Z_AT_DECISION_DIFF',
        'PREV_RELEASE_SPEED', 'PREV_AX', 'PREV_AZ',
        'PREV_X_AT_DECISION', 'PREV_Z_AT_DECISION',
        'STRIKES', 'BALLS',
        'PREV_PLATE_X', 'PREV_PLATE_Z',
        # 타자 관련 features 
        'BATTER_ZONE_SWING', 'BATTER_CHASE_RATE'
    ]

    base_df = rhp[rhp['PITCHER'] != 695243].copy()
    miller_df = rhp[rhp['PITCHER'] == 695243].copy()

    # ── 모델링 ──
    X = base_df[model_features].copy()
    X['STAND'] = (base_df['STAND'] == 'R').astype(int)
    y = base_df['DID_SWING']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    from lightgbm import LGBMClassifier

    lgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=.1,
        num_leaves=31,
        min_child_samples=50,
        subsample=.7,
        reg_alpha=1.0,
        reg_lambda=.1,
        colsample_bytree=.9,
        random_state=42,
        verbose=-1
    )

    lgbm.fit(X_train, y_train)
    y_hat = lgbm.predict(X_test)

    print(f"Train score: {lgbm.score(X_train, y_train):.4f}")
    print(f"Test score:  {lgbm.score(X_test, y_test):.4f}")
    print(metrics.classification_report(y_test, y_hat))


    # ============================================
    # 전이 학습: Mason Miller
    # ============================================

    # ── Miller 데이터 추출 (동일한 전처리 적용) ──
    
    print(f"Miller 전체 투구 수: {len(miller_df)}")
    print(f"Miller 스윙 비율:")
    print(miller_df['DID_SWING'].value_counts(normalize=True))

    # ── Miller 데이터로 fine-tune ──
    X_miller = miller_df[model_features].copy()
    X_miller['STAND'] = (miller_df['STAND'] == 'R').astype(int)
    y_miller = miller_df['DID_SWING']

    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
        X_miller, y_miller, test_size=.3, random_state=42
    )

    # ===================================== 
    # 전이 학습 (base → Miller fine-tune)   
    # =====================================

    fine_model = LGBMClassifier(
        n_estimators=50,        # 적은 트리로 미세 조정
        max_depth=4,
        learning_rate=0.01,     # 작은 보폭
        num_leaves=31,
        min_child_samples=10,   # Miller 데이터가 적으니 완화
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1
    )
    fine_model.fit(X_m_train, y_m_train, init_model=lgbm)
    y_hat_fine = fine_model.predict(X_m_test)
    print("=== 전이 학습 (fine-tune) ===")
    print(f"Score: {fine_model.score(X_m_test, y_m_test):.4f}")
    print(metrics.classification_report(y_m_test, y_hat_fine))

    # ================
    # 단순 블렌딩 기법  
    # ================
    
    base_prob_miller = lgbm.predict_proba(X_m_test)[:, 1]
    fine_prob_miller = fine_model.predict_proba(X_m_test)[:, 1]

    alpha = .4
    blend_prob = (base_prob_miller * alpha) + (fine_prob_miller * (1-alpha))
    blend_pred = (blend_prob >= 0.5).astype(int)

    print("\n=== [최종] 베이스 & 밀러 파인튜닝 블렌딩 앙상블 결과 ===")
    print(f"Blending Test Score: {metrics.accuracy_score(y_m_test, blend_pred):.4f}")
    print(metrics.classification_report(y_m_test, blend_pred))

    # ── 개별 모델 성능 레포트 비교용 출력 ──
    print("=== 모델별 밀러 테스트셋 단독 Accuracy 비교 ===")
    base_invert_score = metrics.accuracy_score(y_m_test, (base_prob_miller >= 0.5).astype(int))
    print(f"1. 순수 베이스 모델이 예측한 밀러 점수: {base_invert_score:.4f}")
    print(f"2. 파인튜닝 단독 모델이 예측한 밀러 점수: {fine_model.score(X_m_test, y_m_test):.4f}")
    print(f"3. 두 모델을 섞은 블렌딩 앙상블 점수: {metrics.accuracy_score(y_m_test, blend_pred):.4f}")
    
    # # ============
    # # 앙상블 기법 
    # # =============
    # from sklearn.ensemble import StackingClassifier, RandomForestClassifier
    # from sklearn.linear_model import LogisticRegression
    # from xgboost import XGBClassifier
    # from lightgbm import LGBMClassifier
    # import time

    # # ── 1차 모델들 (base estimators) ──
    # estimators = [
    #     ('lgbm', LGBMClassifier(
    #         n_estimators=500,
    #         max_depth=4,
    #         learning_rate=0.1,
    #         num_leaves=31,
    #         min_child_samples=50,
    #         subsample=0.7,
    #         colsample_bytree=0.9,
    #         reg_alpha=1.0,
    #         reg_lambda=0.1,
    #         random_state=42,
    #         verbose=-1
    #     )),
    #     ('xgb', XGBClassifier(
    #         n_estimators=300,
    #         max_depth=4,
    #         learning_rate=0.1,
    #         subsample=0.7,
    #         colsample_bytree=0.9,
    #         random_state=42,
    #         verbosity=0
    #     )),
    #     ('rf', RandomForestClassifier(
    #         n_estimators=300,
    #         max_depth=8,
    #         min_samples_leaf=50,
    #         random_state=42,
    #         n_jobs=-1
    #     )),
    # ]

    # # ── 2차 모델 (meta learner) ──
    # stacking = StackingClassifier(
    #     estimators=estimators,
    #     final_estimator=LogisticRegression(max_iter=1000),
    #     cv=3,               # 1차 모델의 예측을 3-Fold CV로 생성
    #     passthrough=False,   # 원본 feature는 2차 모델에 안 넣음
    #     n_jobs=-1,
    #     verbose=1
    # )

    # X = rhp[model_features].copy()
    # X['STAND'] = (rhp['STAND'] == 'R').astype(int)
    # y = rhp['DID_SWING']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    # print("Stacking 학습 시작...")
    # start = time.time()
    # stacking.fit(X_train, y_train)
    # elapsed = time.time() - start
    # print(f"학습 완료: {elapsed/60:.1f}분")

    # y_hat = stacking.predict(X_test)

    # print(f"\nTrain score: {stacking.score(X_train, y_train):.4f}")
    # print(f"Test score:  {stacking.score(X_test, y_test):.4f}")
    # print(metrics.classification_report(y_test, y_hat))

    # # ── 개별 모델과 비교 ──
    # print("\n=== 개별 모델 vs Stacking 비교 ===")
    # for name, model in zip(['lgbm', 'xgb', 'rf'], stacking.estimators_):
    #     score = model.score(X_test, y_test)
    #     print(f"  {name:>6}: {score:.4f}")
    # print(f"  stack: {stacking.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    try:
        conn = DB.connect(user="C##AKSEN", password="1234", dsn="localhost:1521/xe")
        cursor = conn.cursor()

        sql = """
            SELECT *
            FROM TMP_TABLE
        """

        cursor.execute(sql)
        df = pd.DataFrame(
            data= cursor.fetchall(), 
            columns= [d[0] for d in cursor.description]
        )

        # hit_distance_analysis(df)
        # pfx_z_analysis(df)
        # pitch_type_classification(df)
        # Masson_Miller(df)    
        my_model(df)


    except Exception as e:
        traceback.print_exc()
    finally:
        if "conn" in locals():
            conn.close()