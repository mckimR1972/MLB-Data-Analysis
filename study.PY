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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

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

def boosting(df: pd.DataFrame):
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
    from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
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
        boosting(df)

    except Exception as e:
        traceback.print_exc()
    finally:
        if "conn" in locals():
            conn.close()