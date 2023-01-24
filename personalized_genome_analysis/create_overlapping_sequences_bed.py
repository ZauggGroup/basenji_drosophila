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


def create_and_save_seqs(genome_file, needed_chromosomes, seq_len, seqs_bed_file):
    seqs = []

    for record in SeqIO.parse(genome_file, 'fasta'):
        if record.id in needed_chromosomes:
            chr_len = len(record.seq)
            for i in range(0, chr_len, seq_len // 2):
                start_ind = i
                end_ind = start_ind + seq_len
                seqs.append({'chr': record.id, 'start': start_ind, 'end': end_ind})

    write_seqs_bed(seqs_bed_file, seqs, False)


if __name__ == '__main__':
    genome_file = r'/experiments/personalized_predictions/dgp_28/DGRP-28.fa'
    needed_chromosomes = ['chr2L']
    seq_len = 32768
    seqs_bed_file = r'/experiments/personalized_predictions/dgp_28/sequences.bed'
    create_and_save_seqs(genome_file, needed_chromosomes, seq_len, seqs_bed_file)
