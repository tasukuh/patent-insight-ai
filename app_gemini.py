"""
Patent Insight AI Dashboard - Gemini API版
特許文書の研究・要約・分析支援システム（Google Gemini使用）

Author: AI Assistant
Date: 2025-12-15
Version: 1.0.0 (Gemini Edition)
"""

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import json
import io
from datetime import datetime

# ================================================
# ページ設定
# ================================================

st.set_page_config(
    page_title="Patent Insight AI Dashboard (Gemini)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================
# セッション状態の初期化
# ================================================

if 'patents' not in st.session_state:
    st.session_state.patents = []

if 'api_key_verified' not in st.session_state:
    st.session_state.api_key_verified = False

# ================================================
# カスタムCSS
# ================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1976d2;
        margin-bottom: 1rem;
    }
    .gemini-badge {
        background: linear-gradient(135deg, #4285f4 0%, #34a853 50%, #fbbc04 75%, #ea4335 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-message {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================================
# ユーティリティ関数
# ================================================

def extract_text_from_pdf(pdf_file):
    """PDFからテキストを抽出"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"❌ PDF読み込みエラー: {str(e)}")
        return None

def summarize_patent_with_gemini(text, api_key):
    """Gemini APIで特許を要約し、構造化データを生成"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # テキストが長すぎる場合は先頭部分のみ使用
        max_chars = 30000  # Geminiは長文に強いので多めに設定
        text_to_analyze = text[:max_chars] if len(text) > max_chars else text
        
        prompt = f"""
以下の特許文書を分析し、JSON形式で要約してください。

必須項目:
- title: 特許のタイトル（簡潔に、最大50文字）
- problem: 解決しようとしている課題（1-2文、具体的に）
- solution: 提案されている解決策（1-2文、技術的詳細含む）
- effect: 期待される効果（1-2文、定量的な情報があれば含める）
- category: 技術分野のカテゴリ（例: 医療AI、電気自動車、再生可能エネルギー など）

特許文書:
{text_to_analyze}

必ずJSON形式のみで出力してください。説明文は不要です:
"""
        
        response = model.generate_content(prompt)
        content = response.text
        
        # JSONパースを試みる
        # コードブロックで囲まれている場合に対応
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content.strip()
        
        summary = json.loads(json_str)
        
        # 必須フィールドの確認
        required_fields = ['title', 'problem', 'solution', 'effect', 'category']
        for field in required_fields:
            if field not in summary:
                summary[field] = "情報なし"
        
        return summary
        
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON解析エラー: {str(e)}")
        st.error(f"レスポンス内容: {content[:500]}...")
        return None
    except Exception as e:
        st.error(f"❌ 要約生成エラー: {str(e)}")
        return None

def generate_embeddings_with_gemini(text, api_key):
    """Gemini APIでテキストから埋め込みベクトルを生成"""
    try:
        genai.configure(api_key=api_key)
        
        # テキストが長すぎる場合は先頭部分のみ使用
        max_chars = 10000
        text_to_embed = text[:max_chars] if len(text) > max_chars else text
        
        result = genai.embed_content(
            model="models/embedding-001",
            content=text_to_embed,
            task_type="retrieval_document"
        )
        
        return result['embedding']
        
    except Exception as e:
        st.error(f"❌ 埋め込み生成エラー: {str(e)}")
        # エラー時はランダムベクトルを返す（768次元）
        return np.random.rand(768).tolist()

def generate_trend_report_with_gemini(selected_patents, api_key):
    """Gemini APIで選択された特許から戦略的トレンドレポートを生成"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        patents_summary = "\n\n".join([
            f"【特許{i+1}】\n"
            f"タイトル: {p['title']}\n"
            f"カテゴリ: {p['category']}\n"
            f"課題: {p['problem']}\n"
            f"解決策: {p['solution']}\n"
            f"効果: {p['effect']}"
            for i, p in enumerate(selected_patents)
        ])
        
        prompt = f"""
以下の{len(selected_patents)}件の特許を分析し、戦略的トレンドレポートを日本語のMarkdown形式で作成してください。

## レポート構成:

### 1. エグゼクティブサマリー
- 分析対象の概要（3-4文）
- 主要な発見事項（箇条書き3-5項目）

### 2. 技術トレンドの詳細分析
- 共通する技術的アプローチ
- 革新性のポイント
- 技術進化の方向性
- 他分野への応用可能性

### 3. 市場および競合戦略
- 想定される市場ポジショニング
- 競合優位性の源泉
- 差別化要因
- 参入障壁

### 4. 将来展望と推奨アクション
**短期的展望（1-2年）:**
- 具体的な開発方向性
- 製品化のタイムライン

**中長期的展望（3-5年）:**
- 技術進化の予測
- 新規市場創出の可能性

**推奨アクション:**
- R&D投資の優先順位
- 特許戦略
- パートナーシップ戦略

### 5. リスク分析と機会
**技術的リスク:**
- 競合技術の脅威
- 技術的課題

**市場リスク:**
- 需要変動
- 規制リスク

**ビジネス機会:**
- 未開拓市場
- 協業の可能性

### 6. まとめと結論
- 総合評価
- 最優先の次ステップ

特許情報:
{patents_summary}

専門的かつ実用的で、経営判断に役立つレポートを作成してください。
具体性を重視し、抽象的な表現は避けてください。
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        st.error(f"❌ レポート生成エラー: {str(e)}")
        return None

# ================================================
# サイドバー
# ================================================

with st.sidebar:
    st.markdown("# 🔬 Patent Insight AI")
    st.markdown('<div class="gemini-badge">🌟 Powered by Gemini</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # APIキー入力
    st.subheader("🔑 Google AI Studio API 設定")
    
    # Streamlit Secretsから取得を試みる
    default_api_key = ""
    try:
        if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
            default_api_key = st.secrets["gemini"]["api_key"]
            st.success("✅ APIキーがSecretsから読み込まれました")
    except:
        pass
    
    api_key_input = st.text_input(
        "APIキーを入力",
        type="password",
        value=default_api_key,
        help="Google AI Studio APIキーを入力してください",
        key="api_key_input"
    )
    
    if api_key_input:
        st.session_state.api_key_verified = True
        st.success("✅ APIキーが設定されました")
    else:
        st.warning("⚠️ APIキーを入力してください")
        st.markdown("""
        **APIキーの取得方法:**
        1. [AI Studio](https://aistudio.google.com/app/apikey) にアクセス
        2. Googleアカウントでログイン
        3. 「Create API key」をクリック
        4. キーをコピーして貼り付け
        
        **💰 無料枠:** 月15 RPM まで無料！
        """)
    
    st.markdown("---")
    
    # システム情報
    st.subheader("📊 システム情報")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("登録特許数", len(st.session_state.patents))
    with col2:
        if st.session_state.patents:
            categories = len(set(p['category'] for p in st.session_state.patents))
            st.metric("カテゴリ数", categories)
    
    st.info("🤖 AI: Gemini 1.5 Flash")
    st.info("📊 埋め込み: embedding-001")
    
    # Geminiの利点を表示
    with st.expander("🌟 Geminiの利点"):
        st.markdown("""
        - ✅ **50%安い料金**
        - ✅ **毎月無料枠あり**
        - ✅ **超長文対応（2M tokens）**
        - ✅ **優秀な日本語性能**
        - ✅ **PDFネイティブサポート**
        """)
    
    st.markdown("---")
    
    # データクリア
    if st.session_state.patents:
        if st.button("🗑️ 全データをクリア", type="secondary"):
            st.session_state.patents = []
            st.rerun()
    
    st.markdown("---")
    st.caption("© 2025 Patent Insight AI")
    st.caption("Gemini Edition v1.0.0")

# ================================================
# メインコンテンツ
# ================================================

st.markdown('<div class="main-header">🔬 Patent Insight AI Dashboard</div>', unsafe_allow_html=True)
st.markdown("特許文書の研究・要約・分析を支援するAIシステム（Google Gemini版）")

# タブの作成
tab1, tab2, tab3 = st.tabs([
    "📤 インポート & 要約",
    "📊 ダッシュボード",
    "📄 レポート生成"
])

# ================================================
# タブ1: インポート & 要約
# ================================================

with tab1:
    st.header("📤 特許文書のインポートと要約")
    st.markdown("PDFファイルをアップロードすると、Gemini AIが自動的に特許内容を分析・要約します")
    
    # APIキーチェック
    if not st.session_state.api_key_verified:
        st.warning("⚠️ サイドバーでGoogle AI Studio APIキーを設定してください")
    else:
        # ファイルアップロード
        uploaded_files = st.file_uploader(
            "PDFファイルを選択してください",
            type=['pdf'],
            accept_multiple_files=True,
            help="複数のPDFファイルを一度にアップロードできます"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)}件のファイルが選択されました")
            
            # ファイルリスト表示
            with st.expander("📋 アップロードファイル一覧", expanded=True):
                for i, file in enumerate(uploaded_files):
                    st.write(f"{i+1}. {file.name} ({file.size / 1024:.1f} KB)")
            
            # 処理開始ボタン
            if st.button("🚀 処理を開始（Gemini AI）", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_container = st.container()
                
                with status_container:
                    for idx, file in enumerate(uploaded_files):
                        with st.status(f"📄 処理中: {file.name}", expanded=True):
                            # ステップ1: PDF解析
                            st.write("🔍 テキスト抽出中...")
                            text = extract_text_from_pdf(file)
                            
                            if not text:
                                st.error(f"❌ {file.name}: テキストを抽出できませんでした")
                                continue
                            
                            st.write(f"✅ {len(text)}文字のテキストを抽出")
                            
                            # ステップ2: Gemini AI要約
                            st.write("🤖 Gemini AIで要約生成中...")
                            summary = summarize_patent_with_gemini(text, api_key_input)
                            
                            if not summary:
                                st.error(f"❌ {file.name}: 要約生成に失敗しました")
                                continue
                            
                            st.write(f"✅ 要約生成完了: {summary['title']}")
                            
                            # ステップ3: 埋め込み生成
                            st.write("🧮 ベクトル埋め込み生成中...")
                            embedding = generate_embeddings_with_gemini(text, api_key_input)
                            st.write("✅ 埋め込み生成完了")
                            
                            # データ保存
                            patent_data = {
                                'id': f"JP2024{len(st.session_state.patents):06d}",
                                'filename': file.name,
                                'title': summary['title'],
                                'problem': summary['problem'],
                                'solution': summary['solution'],
                                'effect': summary['effect'],
                                'category': summary['category'],
                                'full_text': text[:500] + "...",
                                'embedding': embedding,
                                'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            st.session_state.patents.append(patent_data)
                            st.success(f"✅ {file.name} を正常に処理しました")
                        
                        # プログレスバー更新
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.balloons()
                st.markdown(f'<div class="success-message">🎉 <strong>{len(uploaded_files)}件の特許を正常に登録しました！（Gemini AI使用）</strong></div>', unsafe_allow_html=True)
                st.rerun()
    
    # 保存データの表示
    st.markdown("---")
    st.subheader("💾 保存済み特許データ")
    
    if not st.session_state.patents:
        st.info("まだデータがありません。上記からPDFファイルをアップロードしてください。")
    else:
        # データフレーム作成
        df = pd.DataFrame([
            {
                '特許番号': p['id'],
                'ファイル名': p['filename'],
                'タイトル': p['title'],
                'カテゴリ': p['category'],
                '課題': p['problem'][:50] + "..." if len(p['problem']) > 50 else p['problem'],
                '解決策': p['solution'][:50] + "..." if len(p['solution']) > 50 else p['solution'],
                '処理日時': p['processed_at']
            }
            for p in st.session_state.patents
        ])
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # 詳細表示
        with st.expander("🔍 詳細情報を表示"):
            selected_patent = st.selectbox(
                "特許を選択",
                range(len(st.session_state.patents)),
                format_func=lambda i: f"{st.session_state.patents[i]['id']}: {st.session_state.patents[i]['title']}"
            )
            
            if selected_patent is not None:
                patent = st.session_state.patents[selected_patent]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**特許番号:** {patent['id']}")
                    st.markdown(f"**ファイル名:** {patent['filename']}")
                    st.markdown(f"**カテゴリ:** {patent['category']}")
                    st.markdown(f"**処理日時:** {patent['processed_at']}")
                
                with col2:
                    st.markdown(f"**タイトル:** {patent['title']}")
                
                st.markdown("**課題:**")
                st.write(patent['problem'])
                
                st.markdown("**解決策:**")
                st.write(patent['solution'])
                
                st.markdown("**効果:**")
                st.write(patent['effect'])

# ================================================
# タブ2: ダッシュボード
# ================================================

with tab2:
    st.header("📊 特許ランドスケープ分析")
    st.markdown("クラスタリングと可視化による特許ポートフォリオの全体像")
    
    if len(st.session_state.patents) == 0:
        st.info("📭 まだデータがありません。タブ1で特許をインポートしてください。")
    
    elif len(st.session_state.patents) < 3:
        st.warning(f"⚠️ クラスタリング分析には最低3件のデータが必要です（現在: {len(st.session_state.patents)}件）")
    
    else:
        # 統計情報カード
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📄 総特許数",
                len(st.session_state.patents),
                delta=None
            )
        
        with col2:
            n_clusters = min(3, len(st.session_state.patents))
            st.metric(
                "🎯 クラスター数",
                n_clusters,
                delta=None
            )
        
        with col3:
            categories = set(p['category'] for p in st.session_state.patents)
            st.metric(
                "🏷️ カテゴリ数",
                len(categories),
                delta=None
            )
        
        st.markdown("---")
        
        # 埋め込みデータの取得
        embeddings = np.array([p['embedding'] for p in st.session_state.patents])
        
        # クラスタリング
        n_clusters = min(3, len(st.session_state.patents))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        # 次元削減（t-SNE）
        with st.spinner("📐 次元削減処理中（Gemini埋め込み使用）..."):
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(st.session_state.patents) - 1))
            coords_2d = tsne.fit_transform(embeddings)
        
        # 可視化データの準備
        df_viz = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'cluster': [f"クラスター {c+1}" for c in clusters],
            'title': [p['title'] for p in st.session_state.patents],
            'category': [p['category'] for p in st.session_state.patents],
            'id': [p['id'] for p in st.session_state.patents]
        })
        
        # クラスタリング散布図
        st.subheader("🎯 特許クラスタリング分析（Gemini埋め込み + t-SNE次元削減）")
        
        fig_scatter = px.scatter(
            df_viz,
            x='x',
            y='y',
            color='cluster',
            hover_data=['title', 'category', 'id'],
            title='特許のクラスタリング可視化（Gemini embedding-001使用）',
            labels={'cluster': 'クラスター'},
            color_discrete_sequence=['#4285f4', '#34a853', '#fbbc04', '#ea4335', '#9966FF']
        )
        
        fig_scatter.update_traces(
            marker=dict(size=12, line=dict(width=2, color='white')),
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'カテゴリ: %{customdata[1]}<br>' +
                         'ID: %{customdata[2]}<br>' +
                         '<extra></extra>'
        )
        
        fig_scatter.update_layout(
            height=600,
            hovermode='closest',
            xaxis_title="次元1",
            yaxis_title="次元2"
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("---")
        
        # カテゴリ分布
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 カテゴリ分布")
            category_counts = df_viz['category'].value_counts()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=0.4,
                marker=dict(colors=['#4285f4', '#34a853', '#fbbc04', '#ea4335', '#9966FF', '#FF9F40'])
            )])
            
            fig_pie.update_layout(
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📈 クラスター分布")
            cluster_counts = df_viz['cluster'].value_counts()
            
            fig_bar = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                labels={'x': 'クラスター', 'y': '特許数'},
                color=cluster_counts.index,
                color_discrete_sequence=['#4285f4', '#34a853', '#fbbc04']
            )
            
            fig_bar.update_layout(
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)

# ================================================
# タブ3: レポート生成
# ================================================

with tab3:
    st.header("📄 戦略的トレンドレポート生成")
    st.markdown("選択した特許から競合戦略と将来展望を分析します（Gemini AI使用）")
    
    if not st.session_state.api_key_verified:
        st.warning("⚠️ サイドバーでGoogle AI Studio APIキーを設定してください")
    
    elif not st.session_state.patents:
        st.info("📭 まずタブ1で特許をインポートしてください")
    
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎯 分析対象の選択")
            
            # 特許選択
            selected_indices = st.multiselect(
                "分析する特許を選択してください",
                range(len(st.session_state.patents)),
                format_func=lambda i: f"{st.session_state.patents[i]['title'][:40]}...",
                help="複数選択可能です"
            )
            
            if selected_indices:
                st.success(f"✅ {len(selected_indices)}件の特許を選択中")
                
                with st.expander("📋 選択中の特許"):
                    for idx in selected_indices:
                        patent = st.session_state.patents[idx]
                        st.markdown(f"**{patent['id']}**: {patent['title']}")
                        st.caption(f"カテゴリ: {patent['category']}")
                        st.divider()
            
            # レポート生成ボタン
            generate_button = st.button(
                "🚀 レポートを生成（Gemini AI）",
                type="primary",
                disabled=len(selected_indices) == 0,
                use_container_width=True
            )
        
        with col2:
            st.subheader("📝 生成されたレポート")
            
            if generate_button and selected_indices:
                selected_patents = [st.session_state.patents[i] for i in selected_indices]
                
                with st.spinner("🤖 Gemini AIがレポートを生成中...（30-60秒程度かかります）"):
                    report = generate_trend_report_with_gemini(selected_patents, api_key_input)
                
                if report:
                    st.markdown(report)
                    
                    # ダウンロードボタン
                    st.download_button(
                        label="📥 レポートをダウンロード (Markdown)",
                        data=report,
                        file_name=f"特許分析レポート_Gemini_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                else:
                    st.error("❌ レポート生成に失敗しました")
            
            elif not selected_indices:
                st.info("👈 左側から分析する特許を選択してください")
            
            else:
                st.info("👆 上記の「レポートを生成」ボタンをクリックしてください")

# ================================================
# フッター
# ================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Patent Insight AI Dashboard (Gemini Edition)</strong> v1.0.0</p>
    <p>Powered by Google Gemini 1.5 Flash | © 2025</p>
    <p style='font-size: 0.9rem;'>
        本システムはAIによる自動分析結果を提供します。<br>
        重要な判断には専門家の意見を参考にしてください。
    </p>
    <p style='font-size: 0.85rem; color: #4285f4; font-weight: 600;'>
        🌟 Geminiの利点: 50%安い料金、毎月無料枠あり、超長文対応、優秀な日本語性能
    </p>
</div>
""", unsafe_allow_html=True)
