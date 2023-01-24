import os
import gzip
from Bio import SeqIO


def write_seqs_bed(bed_file, seqs, labels=False):
    """Write sequences to BED file."""
    bed_out = open(bed_file, 'w')
    for i in range(len(seqs)):
        line = '%s\t%d\t%d' % (seqs[i]['chr'], seqs[i]['start'], seqs[i]['end'])
        if labels:
            line += '\t%s' % seqs[i]['label']
        print(line, file=bed_out)
    bed_out.close()


def create_and_save_seqs(genome_files, seq_len, seqs_alt_bed_file, seqs_ref_bed_file,
                         seqs_alt_fasta_file, seqs_ref_fasta_file):
    seqs_alt = []
    seqs_alt_full = []
    seqs_ref = []
    seqs_ref_full = []
    for genome_file in genome_files:
        if not os.path.exists(genome_file):
            print(f'File {genome_file} does not exists')
            continue
        with gzip.open(genome_file, "rt") as file:
            for record in SeqIO.parse(file, 'fasta'):
                chr_len = len(record.seq)
                if record.id.startswith('ALT'):
                    seqs_alt.append({'chr': record.id, 'start': 0, 'end': seq_len})
                    seqs_alt_full.append(record)
                else:
                    seqs_ref.append({'chr': record.id, 'start': 0, 'end': seq_len})
                    seqs_ref_full.append(record)

    write_seqs_bed(seqs_alt_bed_file, seqs_alt, False)
    write_seqs_bed(seqs_ref_bed_file, seqs_ref, False)
    with open(seqs_alt_fasta_file, "w") as file:
        SeqIO.write(sequences=seqs_alt_full, handle=file, format="fasta")
    with open(seqs_ref_fasta_file, "w") as file:
        SeqIO.write(sequences=seqs_ref_full, handle=file, format="fasta")


if __name__ == '__main__':
    for tf in ['bin_1012h', 'bin_68h', 'ctcf_68h', 'mef2_1012h', 'mef2_68h', 'twi_24h']:
        genome_files = [rf'Y:\stojanov\basenji\experiments\data\variants\GATK_raw_{tf}_chr{x}.fa.gz' for x in
                        ['2L', '2R', '3L', '3R', '4']]
        seq_len = 131072
        seqs_alt_bed_file = rf'Y:\stojanov\basenji\experiments\variants_predictions\GATK_raw_{tf}_alt.bed'
        seqs_ref_bed_file = rf'Y:\stojanov\basenji\experiments\variants_predictions\GATK_raw_{tf}_ref.bed'
        seqs_alt_fasta_file = rf'Y:\stojanov\basenji\experiments\variants_predictions\GATK_raw_{tf}_alt.fa'
        seqs_ref_fasta_file = rf'Y:\stojanov\basenji\experiments\variants_predictions\GATK_raw_{tf}_ref.fa'
        create_and_save_seqs(genome_files, seq_len, seqs_alt_bed_file, seqs_ref_bed_file,
                             seqs_alt_fasta_file, seqs_ref_fasta_file)
