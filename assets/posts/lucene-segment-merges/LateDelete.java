import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;

class LateDelete {
    private static final Term REVIEW = new Term("id", "103");

    public static void main(String[] args) throws Exception {
        var mergeWroteOutput = new CountDownLatch(1);
        var deleteApplied = new CountDownLatch(1);
        var config = new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE);
        config.setMergedSegmentWarmer(reader -> {
            try {
                report("Merge output before publication", reader, 3, 4, 4);
                mergeWroteOutput.countDown();
                await(deleteApplied);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IOException(interrupted);
            }
        });

        try (var dir = new ByteBuffersDirectory();
             var writer = new IndexWriter(dir, config);
             var executor = Executors.newSingleThreadExecutor()) {
            addReviews(writer, "101", "102", "103");
            writer.flush();
            addReviews(writer, "104", "105");
            writer.deleteDocuments(new Term("id", "102"));
            writer.commit();

            try (var oldReader = DirectoryReader.open(writer)) {
                report("Original reader before merge", oldReader, 2, 4, 5);
                writer.getConfig().setMergePolicy(new TieredMergePolicy());
                var merge = executor.submit(() -> {
                    writer.forceMerge(1);
                    return null;
                });
                try {
                    await(mergeWroteOutput);
                    // Keep a second merge from reclaiming the late delete before we inspect it.
                    writer.getConfig().setMergePolicy(NoMergePolicy.INSTANCE);
                    writer.deleteDocuments(REVIEW);
                    writer.flush();
                } finally {
                    deleteApplied.countDown();
                }
                merge.get(10, TimeUnit.SECONDS);

                try (var freshReader = DirectoryReader.open(writer);
                     var commitReader = DirectoryReader.open(dir)) {
                    report("Original reader after merge", oldReader, 2, 4, 5);
                    report("Refreshed writer-backed reader", freshReader, -1, 3, 4);
                    report("Directory opened before next commit", commitReader, 2, 4, 5);
                }
                writer.commit();
                try (var commitReader = DirectoryReader.open(dir)) {
                    report("Directory opened after next commit", commitReader, -1, 3, 4);
                }
            }
        }
    }

    private static void addReviews(IndexWriter writer, String... ids) throws IOException {
        for (String id : ids) {
            var doc = new Document();
            doc.add(new StringField("id", id, Field.Store.NO));
            writer.addDocument(doc);
        }
    }

    private static void report(String label, IndexReader reader, int docId, int live, int slots)
            throws IOException {
        var hits = new IndexSearcher(reader).search(new TermQuery(REVIEW), 1).scoreDocs;
        int actualDocId = hits.length == 0 ? -1 : hits[0].doc;
        if (actualDocId != docId || reader.numDocs() != live || reader.maxDoc() != slots) {
            throw new AssertionError(label + ": docID " + actualDocId + ", " + reader.numDocs()
                    + " live / " + reader.maxDoc() + " slots");
        }
        System.out.printf("%-37s review 103 %6s | live %d / slots %d%n", label,
                docId < 0 ? "absent" : docId, live, slots);
    }

    private static void await(CountDownLatch latch) throws InterruptedException, IOException {
        if (!latch.await(10, TimeUnit.SECONDS)) {
            throw new IOException("Timed out waiting for the other thread");
        }
    }
}
